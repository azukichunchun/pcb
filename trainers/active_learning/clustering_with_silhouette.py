import torch
import numpy as np
import random
from collections import Counter
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_samples

from .AL import AL

import os
import pickle

class ClusteringSilhouette(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, n_cand, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device= device 
        self.n_cand = n_cand
        self.silhouette_threshold = 0.05
        
    def run(self, n_query):
        sample_idx = self.rank_uncertainty(n_query)
        return sample_idx

    def rank_uncertainty(self, n_query):
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
            print("| Calculating uncertainty of Unlabeled set")
            for i, data in enumerate(selection_loader):
                inputs = data["img"].to(self.device)
                preds, features = self.model(inputs, get_feature=True)
                preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
                img_features.extend(features.detach().cpu().numpy())
        
        # 画像特徴量をクラスタリング
        img_features = np.array(img_features)
        kmeans = KMeans(n_clusters=self.n_class, random_state=42)
        kmeans.fit(img_features)

        # サンプルごとにsilhouette scoreを計算
        silhouette_scores = silhouette_samples(img_features, kmeans.labels_, metric='euclidean')
        cluster_silhouette_scores = {i: silhouette_scores[kmeans.labels_ == i].mean() for i in range(len(kmeans.cluster_centers_))}

        # 重心から近い順にサンプリング
        distances = pairwise_distances(kmeans.cluster_centers_, img_features) # size: (n_class, n_sample)

        sample_idx = []
        for cluster_id, d in enumerate(distances):
            # d: (1, n_sample)
            if cluster_silhouette_scores[cluster_id] < self.silhouette_threshold: # silhouette scoreが閾値以下のクラスタはランダムサンプル
                continue
            else:
                d_minid = np.argsort(d)
                d_minid = d_minid[0] # 最も近いサンプルのみ選択
                sample_idx.append(d_minid)
        sample_idx_trans = np.array(list(set(sample_idx)))

        assert max(sample_idx_trans) <= len(img_features), f"sample_idx is out of range"

        unselected_idx = [i for i in range(len(img_features))]
        random.shuffle(unselected_idx)

        assert max(sample_idx_trans) <= max(unselected_idx), f"sample_idx is out of range"  

        return sample_idx_trans, unselected_idx


    def select(self, n_query, **kwargs):
        selected_indices, unselected_idx = self.run(n_query)
        Q_index = [self.U_index[idx] for idx in selected_indices]
        Q_index_unselected = [self.U_index[idx] for idx in unselected_idx]

        assert max(Q_index) <= max(Q_index_unselected), f"Q_index is out of range"

        return [Q_index, Q_index_unselected]
