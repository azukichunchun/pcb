import os.path as osp
from random import sample 
import time 
import json
from tqdm import tqdm

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
from .active_learning.pcb import PCB as PCB_ONE_TIME
from .active_learning.pcb_fill import PCB as PCB_FILL
from .active_learning.pcb_silhouette import PCBSilhouette
from .active_learning.badge import BADGE
from .active_learning.coreset import Coreset
from .active_learning.entropy import Entropy
from .active_learning.clustering import Clustering
from .active_learning.clustering_one_sample import ClusteringOneSample
from .active_learning.clustering_with_silhouette import ClusteringSilhouette

_tokenizer = _Tokenizer()



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
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,}

    model = clip.build_model(state_dict or model.state_dict(), design_details)
    
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


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # if not ctx_init.endswith(".json"):
        prompt_prefix = " ".join(["X"] * n_ctx)
        
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

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, desc_file=None):
        super().__init__()
        
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_class_desc=[]
        self.n_cls = len(classnames)
        self.cfg = cfg
        
        if desc_file is not None:
            with open(f"descriptors/descriptors_{desc_file}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())
            classnames = [name.replace("_", " ") for name in classnames]
            for name in classnames:
                name = name.lower()
                self.n_class_desc.append(len(desc_dict[name]))
            
        
    def forward(self, image, get_feature=False):
        image_features = self.image_encoder(image.type(self.dtype))
        
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
                    
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        if self.cfg.TRAINER.COOPAL.AEPATH:
            tmp = []
            start = 0
            for n in self.n_class_desc:
                tmp.append(text_features[start:start+n].mean(dim=0))
                start += n
            text_features = torch.stack(tmp)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        if self.cfg.TRAINER.COOPAL.ASPATH:
            tmp = [] 
            start = 0
            for n in self.n_class_desc:
                tmp.append(torch.sum(logits[:, start:start+n], dim=1)/n)
                start += n
            logits = torch.stack(tmp, dim=1)

        if get_feature:
            return logits, image_features
        else:
            return logits


@TRAINER_REGISTRY.register()
class ALVLM(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.acc = []
        self.query_weights = []
        
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
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(f"prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            #print(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        query_weight = batch["query_weight"]
        input = input.to(self.device)
        label = label.to(self.device)
        query_weight = query_weight.to(self.device)
        return input, label, query_weight

    def load_model(self, directory, epoch=None, round=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        if round is not None:
            model_file = f"{model_file}-{round}"

        for name in names:
            if name == "contrastive":
                continue

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

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            if "contrastive.centers" in state_dict:
                del state_dict["contrastive.centers"]


            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    
    def before_train(self):
        print("INITIALIZE the prompts weights")
        self.build_model()
        
    def after_train(self):
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

    def after_epoch(self, i):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir, model_name=f"model.pth.tar-{self.epoch+1}-{i}")
        
    def train(self):
        """Generic training loops."""
        dataset = build_dataset(self.cfg)
        
        print(f"dataset length: {len(dataset.train_x)}")
        unlabeled_dst = dataset.train_x 
        U_index = list(range(len(unlabeled_dst)))
        if self.cfg.TRAINER.COOP.CSC:
            n_query = dataset.get_num_classes(unlabeled_dst)
        else:
            n_query = dataset.get_num_classes(unlabeled_dst)
        n_cand = int(len(unlabeled_dst) * self.cfg.TRAINER.COOPAL.GAMMA) # 10% of entire dataset
        #n_cand = int(len(unlabeled_dst) * 1.0) # entire dataset
        dataset._train_x = []

        query_impath = []
        for i in range(self.cfg.TRAIN.MAX_ROUND): # クラス数分のデータをサンプルし如何にバランスよくサンプルできるか
            start = time.time()

            if i == 0:
                self.build_model()

            if self.cfg.TRAIN.CURRICULUM and i < self.cfg.TRAIN.STOP_ROUND:
                print(f"{i}-round: Clustering one sample")
                selector = ClusteringOneSample(self.cfg, i, self.model, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), n_cand, True, self.device)
                idx = selector.select(n_cand)
            else:
                print(f"{i}-round: {self.cfg.TRAINER.COOPAL.METHOD} sampling")
                if self.cfg.TRAINER.COOPAL.METHOD == "random":
                    # idx = sample(U_index, n_query)
                    idx = sample(U_index, n_cand)
                elif self.cfg.TRAINER.COOPAL.METHOD == "entropy":
                    selector = Entropy(self.cfg, self.model, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), self.device)
                    idx = selector.select(n_cand)
                elif self.cfg.TRAINER.COOPAL.METHOD == "badge":
                    selector = BADGE(self.cfg, self.model, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), self.device)
                    idx = selector.select(n_cand)
                elif self.cfg.TRAINER.COOPAL.METHOD == "coreset":
                    if i == 0:
                        idx = sample(U_index, n_cand)
                    else:
                        val_x = dataset._train_x.copy()
                        selector = Coreset(self.cfg, self.model, unlabeled_dst, U_index, val_x, dataset.get_num_classes(unlabeled_dst))
                        idx = selector.select(n_cand)
                elif self.cfg.TRAINER.COOPAL.METHOD == "clustering_with_silhouette":
                    selector = ClusteringSilhouette(self.cfg, i, self.model, unlabeled_dst, U_index, dataset.get_num_classes(unlabeled_dst), n_cand, self.device)
                    idx = selector.select(n_cand)
                else:
                    print("NotImplementedError")
                    idx = U_index
            
            #if i != 0:
            statistics = torch.zeros(self.num_classes)
            for elem in dataset._train_x:
                statistics[elem.label] += 1
            if self.cfg.TRAINER.COOPAL.METHOD == "clustering_with_silhouette":
                selector = PCBSilhouette(self.cfg, i, self.model, unlabeled_dst, idx, dataset.get_num_classes(unlabeled_dst), statistics, self.device)
            elif self.cfg.TRAIN.ONE_TIME_SAMPLING:
                selector = PCB_ONE_TIME(self.cfg, i, self.model, unlabeled_dst, idx, dataset.get_num_classes(unlabeled_dst), statistics, self.device)
            else:
                selector = PCB_FILL(self.cfg, self.model, unlabeled_dst, idx, dataset.get_num_classes(unlabeled_dst), statistics, self.device)

            idx = selector.select(n_query)

            # 重みを計算
            if self.cfg.TRAIN.USE_WEIGHTED_LOSS:
                w = 1.0-(1.0-self.cfg.TRAIN.W_LAST) * i/self.cfg.TRAIN.MAX_ROUND
                w = [w] * len(idx)
                #self.query_weights = w + self.query_weights # 1つ前のラウンドのサンプルの重みを減衰させるため手前に結合
                self.query_weights.extend(w)
            else:
                self.query_weights.extend([1.0]*len(idx))

            # Filtering 
            for k in idx:
                dataset._train_x.append(unlabeled_dst[k])
                query_impath.append(unlabeled_dst[k].impath)
                U_index.remove(k)
            
            assert len(self.query_weights) == len(dataset.train_x), f"{len(self.query_weights)} != {len(dataset.train_x)}"
            assert len(U_index) + len(dataset.train_x) == len(unlabeled_dst), f"u index: {len(U_index)}\t train set: {len(dataset.train_x)}\t unlabeled_dst: {len(unlabeled_dst)}"
            
            self.train_loader_x = build_data_loader(
                self.cfg,
                sampler_type=self.cfg.DATALOADER.TRAIN_X.SAMPLER,
                data_source=dataset.train_x,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=True),
                is_train=True,
                dataset_wrapper=None,
                query_weights=self.query_weights
            )
            # self.model.train()
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.run_epoch()
                self.after_epoch(i)
            self.after_train()
            print("training time for {}-th round: {:.2f} seconds".format(i, time.time() - start))
        print("=== Result Overview ===")
        for i in range(len(self.acc)):
            print(f"{i}: {self.acc[i]}")
        print("=======================") 

        with open(f"{self.cfg.OUTPUT_DIR}/query_impath.txt", "w") as f:
            for impath in query_impath:
                f.write(impath + "\n")

    @torch.no_grad()
    def test_new(self, model_dir, epoch, split=None):

        for i in range(self.cfg.TRAIN.MAX_ROUND):
            self.build_model()
            self.load_model(model_dir, epoch=epoch, round=i)

            """A generic testing pipeline."""
            self.set_model_mode("eval")
            self.evaluator.reset()

            if split is None:
                split = self.cfg.TEST.SPLIT

            if split == "val" and self.val_loader is not None:
                data_loader = self.val_loader
            else:
                split = "test"  # in case val_loader is None
                data_loader = self.test_loader

            print(f"Evaluate on the *{split}* set")

            for batch_idx, batch in enumerate(tqdm(data_loader)):
                input, label = self.parse_batch_test(batch)
                output = self.model(input)
                self.evaluator.process(output, label)

            results = self.evaluator.evaluate()

            for k, v in results.items():
                tag = f"{split}/{k}"
                self.write_scalar(tag, v, self.epoch)
            self.acc.append(list(results.values())[0])

        print("=== Result Overview ===")
        for i in range(len(self.acc)):
            print(f"{i}: {self.acc[i]}")
        print("=======================") 

        return
