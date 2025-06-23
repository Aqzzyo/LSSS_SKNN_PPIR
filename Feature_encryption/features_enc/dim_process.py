from typing import List

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize



from pyretri.index.dim_processor.dim_processors_impl import l2_normalize, pca
from pyretri.index.utils.feature_loader import FeatureLoader

# def DimEncfea(query_fea, gallery_fea):
#     """
#         Args:
#             query_fea (np.ndarray): query set features.
#             gallery_fea (np.ndarray): gallery set features.
#         Returns:
#             query features and gallery features after process.
#         """
#     pca = PCA(n_components=128, whiten=False)
#     query_fea_l2 = normalize(query_fea, norm="l2")
#     gallery_fea_l2 = normalize(gallery_fea, norm="l2")
#     # pca.fit(gallery_fea_l2)
#     #
#     # query_fea_pca = pca.transform(query_fea)
#     # gallery_fea_pca = pca.transform(gallery_fea)
#
#     query_fea_pca = pca.fit_transform(query_fea_l2)
#     gallery_fea_pca = pca.fit_transform(gallery_fea_l2)
#     print(pca.explained_variance_ratio_)
#
#     query_fea_pca_l2 = normalize(query_fea_pca, norm="l2")
#     gallery_fea_pca_l2 = normalize(gallery_fea_pca, norm="l2")
#
#     return query_fea_pca_l2, gallery_fea_pca_l2

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def DimEncfea(query_fea, gallery_fea):
    # 合并数据统一处理
    all_fea = np.concatenate([query_fea, gallery_fea], axis=0)

    # scaler = StandardScaler()
    # all_fea_sca = scaler.fit_transform(all_fea)
    all_fea_l2 = normalize(all_fea, norm='l2')

    # 自动选择保留95%方差的主成分数量
    pca = PCA(n_components=64, whiten=False)
    all_fea_pca = pca.fit_transform(all_fea_l2)

    # 拆分回query和gallery
    split_idx = len(query_fea)
    query_fea_pca = all_fea_pca[:split_idx]
    gallery_fea_pca = all_fea_pca[split_idx:]

    # L2归一化
    query_fea_pca_l2 = normalize(query_fea_pca, norm='l2')
    gallery_fea_pca_l2 = normalize(gallery_fea_pca, norm='l2')

    # 打印方差保留信息
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"主成分数量: {pca.n_components_}")
    print(f"累积方差占比: {cum_var[-1]:.4f} ({cum_var[-1] * 100:.2f}%)")

    return query_fea_pca_l2, gallery_fea_pca_l2