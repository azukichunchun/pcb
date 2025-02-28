import torch
import numpy as np
import random
from scipy import stats
from collections import Counter
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_samples

from .AL import AL

import os
import pickle

class ClusteringSilhouetteBadge(AL):
    def __init__(self, cfg, i, model, unlabeled_dst, U_index, n_class, n_cand, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device= device 
        self.n_cand = n_cand
        self.round = i
        self.silhouette_threshold = 0.05
        
    def run(self, n_query):
        sample_idx = self.rank_uncertainty(n_query)
        return sample_idx

    def rank_uncertainty(self, n_query):
        self.model.eval()
        if self.cfg.MODEL.BACKBONE.NAME != "RN50":
            embDim = 512
        else:
            embDim = 1024
        num_unlabeled = len(self.U_index)
        grad_embeddings = torch.zeros([num_unlabeled, embDim * self.n_class])
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
            print("|| Calculating badge embeddings")
            for i, data in enumerate(selection_loader):
                inputs = data["img"].to(self.device)
                preds, features = self.model(inputs, get_feature=True)
                preds = torch.nn.functional.softmax(preds, dim=1).data
                maxInds = torch.argmax(preds, 1)
                img_features.extend(features.detach().cpu().numpy())
        
                for j in range(len(inputs)):
                    for c in range(self.n_class):
                        if c == maxInds[j]:
                            grad_embeddings[i * len(inputs) + j][embDim * c: embDim * (c + 1)] = features[j].clone() * (
                                        1 - preds[j][c])
                        else:
                            grad_embeddings[i * len(inputs) + j][embDim * c: embDim * (c + 1)] = features[j].clone() * (
                                        -1 * preds[j][c])

        # badge基準でサンプルを選択
        selected_indices_badge = self.k_means_plus_centers(X=grad_embeddings.cpu().numpy(), K=n_query)

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
                print(f"Cluster {cluster_id} is not selected because silhouette score is lower than threshold")
                continue
            else:
                # if self.round == 0:
                #     print("First round: Sampling the nearest sample to the centroid")
                #     d_minid = np.argsort(d)
                #     d_minid = d_minid[0] # 最も近いサンプルのみ選択
                #     sample_idx.append(d_minid)
                # else:
                print("Subsequent rounds: Sampling the random samples in each cluster")
                sample_idx_in_cluster = np.where(kmeans.labels_ == cluster_id)[0] # クラスタ内のサンプルインデックス
                sample_idx_in_cluster_badge = [i for i in sample_idx_in_cluster if i in selected_indices_badge] # badgeで選択されたサンプルのみ
                if len(sample_idx_in_cluster_badge) == 0:
                    sample_idx.append(random.choice(sample_idx_in_cluster)) # badgeで選択されたサンプルがない場合はクラスタ内でランダムサンプル
                else:
                    sample_idx.append(random.choice(sample_idx_in_cluster_badge))
        sample_idx_trans = np.array(list(set(sample_idx)))

        assert max(sample_idx_trans) <= len(img_features), f"sample_idx is out of range"

        unselected_idx = [i for i in range(len(img_features))]
        random.shuffle(unselected_idx)

        assert max(sample_idx_trans) <= max(unselected_idx), f"sample_idx is out of range"  

        return sample_idx_trans, unselected_idx

    # kmeans ++ initialization
    def k_means_plus_centers(self, X, K):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)

                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]

            if len(mu) % 100 == 0:
                print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
                
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]

            while ind in indsAll: 
                ind = customDist.rvs(size=1)[0]

            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def select(self, n_query, **kwargs):
        selected_indices, unselected_idx = self.run(n_query)
        Q_index = [self.U_index[idx] for idx in selected_indices]
        Q_index_unselected = [self.U_index[idx] for idx in unselected_idx]

        assert max(Q_index) <= max(Q_index_unselected), f"Q_index is out of range"

        return [Q_index, Q_index_unselected]
