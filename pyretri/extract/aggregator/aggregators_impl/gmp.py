# -*- coding: utf-8 -*-

import torch
from sklearn.decomposition import PCA

from ..aggregators_base import AggregatorBase
from ...registry import AGGREGATORS

from typing import Dict

@AGGREGATORS.register
class GMP(AggregatorBase):
    """
    Global maximum pooling
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        self.first_show = True
        super(GMP, self).__init__(hps)

    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        ret = dict()
        pca = PCA(n_components=128)
        for key in features:
            fea = features[key]
            if fea.ndimension() == 4:
                fea = (fea.max(dim=3)[0]).max(dim=2)[0]
                fea_pac = fea.cpu()
                fea_pac = fea_pac.numpy()
                fea_cpu = pca.fit_transform(fea_pac)
                fea = torch.from_numpy(fea_cpu).to('cuda:0')
                ret[key + "_{}".format(self.__class__.__name__)] = fea
            else:
                # In case of fc feature.
                assert fea.ndimension() == 2
                if self.first_show:
                    print("[GMP Aggregator]: find 2-dimension feature map, skip aggregation")
                    self.first_show = False
                ret[key] = fea

        return ret
