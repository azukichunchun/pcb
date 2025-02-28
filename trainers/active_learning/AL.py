import torch

class AL(object):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, **kwargs):
        self.unlabeled_dst = unlabeled_dst
        self.cluster_centers = None
        if len(U_index)==2: # クラスタリング中心IDとそれ以外のIDのリスト
            self.U_index = U_index[1]
            try:
                self.cluster_centers = [self.U_index.index(idx) for idx in U_index[0]]
            except ValueError:
                import pdb; pdb.set_trace()
            assert max(self.cluster_centers) <= max(self.U_index), f"cluster_centers is out of range"
        else:
            self.U_index = U_index
        self.unlabeled_set = torch.utils.data.Subset(unlabeled_dst, self.U_index)
        if self.cluster_centers is not None:
            assert max(self.cluster_centers) <= len(self.unlabeled_set), f"cluster_centers is out of range"
        self.n_unlabeled = len(self.unlabeled_set)
        self.n_class = n_class
        self.model = model
        self.index = []
        self.cfg = cfg

    def select(self, **kwargs):
        return