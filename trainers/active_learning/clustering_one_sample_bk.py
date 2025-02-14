import torch
import numpy as np
import random
from collections import Counter
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from .AL import AL

import os
import pickle

class ClusteringOneSample(AL):
    def __init__(self, cfg, i, model, unlabeled_dst, U_index, n_class, n_cand, sort_by_ent, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device= device 
        self.n_cand = n_cand
        self.sort_by_ent = sort_by_ent
        self.round = i
        self.n_shell = self.cfg.TRAIN.MAX_ROUND
        self.shell_bound = 33 # percentile
        self.curriculum = self.cfg.TRAIN.CURRICULUM

    def run(self, n_query):
        sample_idx = self.rank_uncertainty()
        #selection_result = np.argsort(scores)[:n_query]
        return sample_idx
    
    def rank_uncertainty(self):
        self.model.eval()
        with torch.no_grad():
            # selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
            selection_loader = build_data_loader(
                self.cfg, 
                data_source=self.unlabeled_set, 
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )

            img_features = []
            entropies = []
            print("| Calculating uncertainty of Unlabeled set")
            for i, data in enumerate(selection_loader):
                inputs = data["img"].to(self.device)
                preds, features = self.model(inputs, get_feature=True)
                preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
                entropys = (np.log(preds + 1e-6) * preds).sum(axis=1)
                img_features.extend(features.detach().cpu().numpy())
                entropies.extend(entropys)
        
        # 画像特徴量をクラスタリング
        img_features = np.array(img_features)
        kmeans = KMeans(n_clusters=self.n_class, random_state=42)
        kmeans.fit(img_features)

        # 重心から近い順にサンプリング
        distances = pairwise_distances(kmeans.cluster_centers_, img_features) # size: (n_class, n_sample)

        sample_idx = []
        for d in distances:
            if self.curriculum:
                current_bound = self.shell_bound * (self.round / self.n_shell)
                d_bound = np.percentile(d, current_bound)
                print(f'current_bound: {current_bound}, d_bound: {d_bound}')
                d_sort = np.sort(d)
                d_argsort = np.argsort(d)

                bound_idx = np.where(d_sort >= d_bound)[0][0]
                d_minid = d_argsort[bound_idx]
                sample_idx.append(d_minid)
            else:
                d_minid = np.argsort(d)
                d_minid = d_minid[0] # 最も近いサンプルのみ選択
                sample_idx.append(d_minid)
        sample_idx_trans = np.array(list(set(sample_idx)))

        # サンプル数がクラスサイズと異なる場合は追加でランダムにデータを取る
        while len(sample_idx_trans) != self.n_class:
            print(f'Size is differenct. sample size: {len(sample_idx_trans)}, n_class: {self.n_class}')
            added_sample_size = int(self.n_class - len(sample_idx_trans))
            nums = random.sample(range(0, len(img_features)-1), added_sample_size)
            sample_idx_trans = np.concatenate([sample_idx_trans, nums])
        return sample_idx_trans

    def select(self, n_query, **kwargs):
        selected_indices = self.run(n_query)
        
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index
