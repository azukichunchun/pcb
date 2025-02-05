import os.path as osp
from random import sample 
from collections import OrderedDict
import time 
import json 

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.datasets import build_dataset
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
#from .active_learning.pcb import PCB
from .active_learning.pcb_fill import PCB
from .active_learning.badge import BADGE
from .active_learning.coreset import Coreset
from .active_learning.entropy import Entropy
from .active_learning.clustering import Clustering
from .alvlm import ALVLM
from contrastive import Proximity, Con_Proximity

import pdb

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class TextEncoder_Orig(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype
        

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # if not ctx_init.endswith(".json"):
        prompt_prefix = " ".join(["X"] * n_ctx)
        
        # template
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        temp_prompts = [temp.format(c.replace("_", "")) for c in classnames]
        temp_prompts = torch.cat([clip.tokenize(p) for p in temp_prompts])
        self.temp_prompts = temp_prompts
        self.temp_prompts.to(self.device)
        
        classnames = [name.replace("_", " ") for name in classnames]
        n_desc_per_cls = None
        if cfg.TRAINER.COOPAL.ASPATH:
            with open(f"descriptors/descriptors_{cfg.TRAINER.COOPAL.ASPATH}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
                
            name_lens, prompts = [], []
            for name in classnames:
                name = name.lower()
                for desc in desc_dict[name]:
                    name_lens.append(len(_tokenizer.encode(f"{name}, which is/has {desc}")))
                    prompts.append(prompt_prefix + " " + f"{name}, which is/has {desc}.")
                    
        elif cfg.TRAINER.COOPAL.AEPATH:
            with open(f"descriptors/descriptors_{cfg.TRAINER.COOPAL.AEPATH}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
                
            name_lens, prompts = [], []
            for name in classnames:
                name = name.lower()
                for desc in desc_dict[name]:
                    name_lens.append(len(_tokenizer.encode(f"{name}, which is/has {desc}")))
                    prompts.append(prompt_prefix + " " + f"{name}, which is/has {desc}.")
                    
        else:
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print(prompts)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
       
       
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = embedding.size(0)
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
       
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(self.n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        self.vis_dim = vis_dim

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()


    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts    


    def forward(self, im_features=None):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, desc_file=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.text_encoder_orig = TextEncoder_Orig(clip_model)
        self.classnames = classnames
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_class_desc=[]
        self.n_cls = len(classnames)
        self.vis_dim =self.prompt_learner.vis_dim
        self.cfg = cfg

        if cfg.DATASET.SUBSAMPLE_CLASSES == "new":
            if cfg.DATASET.NUM_CLASS % 2 != 0:
                num_class_training = self.n_cls + 1
            else:
                num_class_training = self.n_cls
        else:
            num_class_training = self.n_cls

        self.criterion_conprox = Con_Proximity(num_classes=num_class_training, 
                                               feat_dim=self.vis_dim,
                                               dtype=self.dtype,
                                               use_gpu=torch.cuda.is_available())

        self.criterion_prox = Proximity(num_classes=num_class_training, 
                                        feat_dim=self.vis_dim,
                                        dtype=self.dtype,
                                        use_gpu=torch.cuda.is_available())

        if desc_file is not None:
            with open(f"descriptors/descriptors_{desc_file}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
            classnames = [name.replace("_", " ") for name in classnames]
            for name in classnames:
                name = name.lower()
                self.n_class_desc.append(len(desc_dict[name]))


    def forward(self, image, label=None, get_feature=False, use_template=False):

        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
        prompts = self.prompt_learner(image_features)
        
        logits = []
        text_features = []
        for pts_i, imf_i in zip(prompts, image_features):
            txf = self.text_encoder(pts_i, tokenized_prompts)
            text_features.append(txf)
            txf = txf / txf.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ txf.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        text_features = torch.stack(text_features)

        if get_feature:
            return logits, image_features
        
        elif use_template:
            with torch.no_grad():
                temp_prompts = self.prompt_learner.temp_prompts.to(self.device)
                text_features = self.text_encoder_orig(temp_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = logit_scale * image_features @ text_features.t()
                return logits

        elif self.prompt_learner.training:

            cross_entropy_loss = F.cross_entropy(logits, label)

            # Inclusive and Exclusive Loss
            conprox_loss_list = []
            prox_loss_list = []
            text_labels = range(len(self.classnames))
            text_labels = torch.tensor(text_labels).to(self.device)
            for txf in text_features:
                conprox_loss = self.criterion_conprox(txf, text_labels)
                conprox_loss_list.append(conprox_loss)

                prox_loss = self.criterion_prox(torch.cat((image_features, txf)), 
                                                torch.cat((label, text_labels)))
                prox_loss_list.append(prox_loss)
            conprox_loss = min(conprox_loss_list)
            prox_loss = max(prox_loss_list)

            return cross_entropy_loss, conprox_loss, prox_loss
        
        else:
            return logits


@TRAINER_REGISTRY.register()
class ALVLM_DoCoCoOp(ALVLM):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.acc = []
        
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        if cfg.TRAINER.COOPAL.ASPATH:
            self.model = CustomCLIP(cfg, classnames, clip_model, desc_file=cfg.TRAINER.COOPAL.ASPATH)
        elif cfg.TRAINER.COOPAL.AEPATH:
            self.model = CustomCLIP(cfg, classnames, clip_model, desc_file=cfg.TRAINER.COOPAL.AEPATH)
        else:
            self.model = CustomCLIP(cfg, classnames, clip_model)
        #print(self.model)
        
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name or "criterion" in name:
                param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")


        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        self.optim_conprox = build_optimizer(self.model.criterion_conprox, self.cfg.OPTIM_CONPROX)
        self.optim_prox = build_optimizer(self.model.criterion_prox, self.cfg.OPTIM_PROX)
        self.sched_conprox = build_lr_scheduler(self.optim_conprox, self.cfg.OPTIM_CONPROX)
        self.sched_prox = build_lr_scheduler(self.optim_prox, self.cfg.OPTIM_PROX)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model(f"prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("conprox", self.model.criterion_conprox, self.optim_conprox, self.sched_conprox)
        self.register_model("prox", self.model.criterion_prox, self.optim_prox, self.sched_prox)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)
        #     #print(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        model = self.model
        optim = self.optim
        scaler = self.scaler
        optim_prox = self.optim_prox
        optim_conprox = self.optim_conprox
        sched = self.sched
        sched_prox = self.sched_prox
        sched_conprox = self.sched_conprox
        
        prec = self.cfg.TRAINER.COOP.PREC

        cross_entropy_loss, conprox_loss, prox_loss = model(image, label)

        if self.epoch <= self.cfg.TRAIN.PROX_START_EPOCH:           
            loss = cross_entropy_loss                
            optim.zero_grad() 
            loss.backward()
            optim.step()

            if (self.batch_idx + 1) == self.num_batches:
                sched.step()
        else:
            loss = cross_entropy_loss + self.cfg.TRAINER.DOCOCOOPAL.LAMBDA_PROX * prox_loss - self.cfg.TRAINER.DOCOCOOPAL.LAMBDA_CONPROX * conprox_loss

            optim.zero_grad()
            optim_conprox.zero_grad()
            optim_prox.zero_grad()

            loss.backward()
            optim.step()

            for param in self.model.criterion_conprox.parameters():
                param.grad.data *= (1. / self.cfg.TRAINER.DOCOCOOPAL.LAMBDA_CONPROX)

            for param in self.model.criterion_prox.parameters():
                param.grad.data *= (1. / self.cfg.TRAINER.DOCOCOOPAL.LAMBDA_PROX)

            optim_conprox.step()
            optim_prox.step()

            if (self.batch_idx + 1) == self.num_batches:
                sched.step()
                sched_prox.step()
                sched_conprox.step()

        loss_summary = {
            "loss(ce)": cross_entropy_loss.item(),
            "loss(prox)": prox_loss.item(),
            "loss(conprox)": conprox_loss.item()
        }

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"
        epoch = self.cfg.OPTIM.MAX_EPOCH

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    
    def before_train(self):
        print("INITIALIZE the prompts weights")
        self.build_model()
        
    def after_train(self):
        print("Finish training")
        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.acc.append(self.test())
            
        # Close writer
        self.close_writer()