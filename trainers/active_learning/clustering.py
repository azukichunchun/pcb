import torch
import numpy as np
from collections import Counter
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from .AL import AL

class Clustering(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, n_cand, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device= device 
        self.n_cand = n_cand

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
            scores = np.array([])
            img_features = []
            
            print("| Calculating uncertainty of Unlabeled set")
            for i, data in enumerate(selection_loader):
                inputs = data["img"].to(self.device)
                
                preds, features = self.model(inputs, get_feature=True)
                preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
                entropys = (np.log(preds + 1e-6) * preds).sum(axis=1)
                scores = np.append(scores, entropys)
                
                img_features.extend(features.detach().cpu().numpy())
        
        # 画像特徴量をクラスタリング
        img_features = np.array(img_features)
        kmeans = KMeans(n_clusters=self.n_class, random_state=42)
        kmeans.fit(img_features)

        # 重心から近い順にサンプリング
        distances = pairwise_distances(kmeans.cluster_centers_, img_features)
        cluster_counts = Counter(kmeans.labels_)
        sample_idx = []
        for cluster_id, d in enumerate(distances):
            d_minid = np.argsort(d)
            d_minid = d_minid[:int(cluster_counts[cluster_id] * self.cfg.TRAINER.COOPAL.GAMMA)]
            sample_idx.append(d_minid)
        sample_idx = np.array(sample_idx)
        sample_idx = np.transpose(sample_idx)
        sample_idx = np.concatenate(sample_idx)
        sample_idx = np.array(list(set(sample_idx)))

        return sample_idx

    def select(self, n_query, **kwargs):
        selected_indices = self.run(n_query)
        import pdb; pdb.set_trace()
        
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index
