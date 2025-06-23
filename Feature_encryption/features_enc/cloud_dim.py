from typing import List

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import sys
sys.path.append('/root/autodl-tmp/pyretri/')

from pyretri.index.dim_processor.dim_processors_impl import l2_normalize, pca
from pyretri.index.utils.feature_loader import FeatureLoader

def DimEncfea(query_fea, gallery_fea):
    """
        Args:
            query_fea (np.ndarray): query set features.
            gallery_fea (np.ndarray): gallery set features.
        Returns:
            query features and gallery features after process.
        """
    pca = PCA(n_components=512, whiten=False)
    query_fea_l2 = normalize(query_fea, norm="l2")
    gallery_fea_l2 = normalize(gallery_fea, norm="l2")
    pca.fit(gallery_fea_l2)

    query_fea_pca = pca.transform(query_fea)
    gallery_fea_pca = pca.transform(gallery_fea)

    query_fea_pca_l2 = normalize(query_fea_pca, norm="l2")
    gallery_fea_pca_l2 = normalize(gallery_fea_pca, norm="l2")

    return query_fea_pca_l2, gallery_fea_pca_l2

