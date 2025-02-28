import torch
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
import random
import pickle
import copy

from .AL import AL


class PCBSilhouette(AL):
    def __init__(self, cfg, i, model, unlabeled_dst, U_index, n_class, statistics, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device 
        self.pred = []
        self.statistics = statistics 
        self.round = i
        self.cfg = cfg

    def select(self, n_query, **kwargs):
        self.pred = []
        self.model.eval()
        # embDim = self.model.image_encoder.attnpool.c_proj.out_features
        num_unlabeled = len(self.U_index)
        assert len(self.unlabeled_set) == num_unlabeled, f"{len(self.unlabeled_dst)} != {num_unlabeled}"
        with torch.no_grad():
            unlabeled_loader = build_data_loader(
                self.cfg,
                data_source=self.unlabeled_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )

            # generate entire unlabeled features set
            for i, batch in enumerate(unlabeled_loader):
                inputs = batch["img"].to(self.device)
                out = self.model(inputs, get_feature=False)
                batchProbs = torch.nn.functional.softmax(out, dim=1).data
                maxInds = torch.argmax(batchProbs, 1)
                self.pred.append(maxInds.detach().cpu())
        self.pred = torch.cat(self.pred)
        
        # クラスタ中心で選んだものはpredに関係なくQ_indexに加える
        Q_index = []
        true_class_counts = copy.deepcopy(self.statistics)
        Q_index.extend(self.cluster_centers)
        
        assert max(self.cluster_centers) <= len(self.unlabeled_set), f"{max(self.cluster_centers)} > {len(self.unlabeled_set)}"

        # クラスタ中心の真のラベルをもとにself.statisticsを更新
        for idx in self.cluster_centers:
            self.statistics[self.unlabeled_set[idx].label] += 1
            true_class_counts[self.unlabeled_set[idx].label] += 1
        
        # その後はpredに従い、まだ選ばれていないクラスのものをQ_indexに加える
        while len(Q_index) < n_query:
            min_cls = int(torch.argmin(self.statistics))
            sub_pred = (self.pred == min_cls).nonzero().squeeze(dim=1).tolist()
            if len(sub_pred) == 0:
                num = random.randint(0, num_unlabeled-1)
                while num in Q_index or num in self.cluster_centers:
                        num = random.randint(0, num_unlabeled-1)
                if num not in Q_index:  # 追加前に再チェック
                    Q_index.append(num)
            else:
                random.shuffle(sub_pred)
                for num in sub_pred:
                    if num not in self.cluster_centers: # クラスタ中心は選ばない
                        if num not in Q_index: # すでに選ばれるものは選ばない
                            Q_index.append(num)
                            self.statistics[min_cls] += 1
                            break 
                else: 
                    num = random.randint(0, num_unlabeled-1)
                    while num in Q_index or num in self.cluster_centers:
                            num = random.randint(0, num_unlabeled-1)
                    if num not in Q_index:            
                        Q_index.append(num)
            true_class_counts[self.unlabeled_set[num].label] += 1
            print(true_class_counts)

        Q_index = [self.U_index[idx] for idx in Q_index]

        assert len(Q_index) == len(set(Q_index)), f"Q_index contains duplicate indices"

        return Q_index