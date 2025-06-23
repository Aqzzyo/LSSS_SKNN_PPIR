
import concurrent.futures
import numpy as np
import multiprocessing
from pyretri.index.utils.feature_loader import FeatureLoader

import random
from tqdm import tqdm
from index_gen import index_gen as IG
import general_utils as util
import json
from keyGen import readKeyFromFile
from dim_process import DimEncfea
import time
featureLoader = FeatureLoader()
import logging

logging.basicConfig(level=logging.INFO)
import random
import numpy as np


def generate_random_vector(dim, min_val=0, max_val=1, dtype='float'):
    """
    生成指定维度的随机向量
    返回:
    包含随机数的列表
    """
    # 验证维度有效性
    if not isinstance(dim, int) or dim < 1:
        raise ValueError("维度必须是正整数")

    # 确保min_val和max_val有效
    if max_val < min_val:
        raise ValueError("max_val不能小于min_val")

    # 处理特殊情况
    if min_val == max_val:
        return [min_val] * dim

    # 生成随机向量
    if dtype == 'int':
        return [random.randint(int(min_val), int(max_val)) for _ in range(dim)]
    elif dtype == 'float':
        return [random.uniform(min_val, max_val) for _ in range(dim)]
    else:
        raise ValueError("dtype必须是'int'或'float'")


import numpy as np


def np_integer_noise_vector(length, min_val=-5, max_val=5, distribution='uniform', scale=1.0):
    # 参数验证（同上）

    if min_val == max_val:
        return np.full(length, min_val)

    if distribution == 'uniform':
        noise = np.random.randint(min_val, max_val + 1, length)

    elif distribution == 'normal':
        noise_fl = np.random.normal(0, scale, length)
        noise = np.clip(noise_fl, min_val, max_val).astype(int)

    elif distribution == 'laplace':
        noise_fl = np.random.laplace(0, scale / np.sqrt(2), length)
        noise = np.clip(noise_fl, min_val, max_val).astype(int)

    else:
        raise ValueError("分布必须是'uniform', 'normal'或'laplace'")

    return noise.tolist()

def readFeatures(filename):

    query_fea_dir = f"E:/Github/PyRetri-master/data/features/PrecisionTest/75_1GeM/{filename}"
    gallery_fea_dir = f"E:/Github/PyRetri-master/data/features/PrecisionTest/75_1GeM/{filename}"

    feature_names = ['pool5_GeM']

    if filename == "query":
        query_fea, query_info, _ = featureLoader.load(query_fea_dir, feature_names)
        return query_fea, query_info
    if filename == "gallery":
        gallery_fea, gallery_info, _ = featureLoader.load(gallery_fea_dir, feature_names)
        return gallery_fea, gallery_info


def indexGen(feature_vec, key):

    dimension = feature_vec.shape[1]
    M1 = key["M1"]
    M2 = key["M2"]
    permutation = key["permutation"]
    #choose randm num between 1-100.
    # alpha = random.randint(1, 100)
    # rf = random.randint(1, 100)
    alpha = 1
    rf = 1
    ext_feature_vec = IG.extend_feature_vector(feature_vec, alpha, rf)
    ext_feature_vec = np.array(ext_feature_vec)
    permuted_fv = util.permute_feature_vector(ext_feature_vec, permutation)

    diagonal_elements = np.array(permuted_fv)


    F = np.diag(diagonal_elements).astype('float64')
    # F = util.create_diagonal_matrix(permuted_fv)
    S = util.generate_lower_triangular_matrix(dimension)
    E = util.multiply_matrices_gpu([M1, S, F, M2])

    return E
def index_DVREI(feature_vec,key):
    start1 = time.perf_counter()
    dimension = feature_vec.shape[1]
    M = key["M"]
    M_inv = key["M_inv"]
    lamda = key["lamda"]
    beta = generate_random_vector(511)
    yibux = np_integer_noise_vector(1024)
    betaq = 40
    ext_feature_vec = IG.extend_feature_vector_DVREI(feature_vec, betaq, beta)
    ext_feature_vec_final = IG.extend_feature_vector_DVREI_T(ext_feature_vec, lamda, yibux)
    ext_feature_vec_final = np.array(ext_feature_vec_final)
    fq = np.dot(M_inv, ext_feature_vec_final)
    end1 = time.perf_counter()
    # print(end1 - start1)
    return  fq




def indexGenQ(feature_vec, key):
    start1 = time.perf_counter()
    dimension = feature_vec.shape[1]
    M1_inv = key["M1_inv"]
    M2_inv = key["M2_inv"]
    permutation = key["permutation"]
    #choose randm num between 1-100.
    # beta = random.randint(1, 100)
    # rq = random.randint(1, 100)
    beta = 1
    rq = 1
    sim = 0.5
    c = 1
    theta = IG.convert_sim_to_theta(sim, c)

    ext_feature_vec = IG.extend_feature_vector_Q(feature_vec, beta, rq, theta)
    ext_feature_vec = np.array(ext_feature_vec)
    permuted_fv = util.permute_feature_vector(ext_feature_vec, permutation)

    diagonal_elements = np.array(permuted_fv)
    Q = np.diag(diagonal_elements).astype('float64')
    # Q = util.create_diagonal_matrix(permuted_fv)
    S = util.generate_lower_triangular_matrix(dimension)

    D = util.multiply_matrices_gpu([M2_inv, Q, S, M1_inv])
    end1 = time.perf_counter()
    # print(end1 - start1)

    return D

# def process_feature(idx, feature, key):
#     return {"id": idx, "E": indexGen(feature, key).tolist()}

def createIndex(features, key):

    results = []
    i = 0
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for feature in tqdm(features):
        start1 = time.perf_counter()
        temp = {}
        id = i
        feature_vec = feature

        E = indexGenQ(feature_vec, key)

        temp["id"] = id
        temp["Q"] = E.tolist()
        i = i+1

        # results.append(temp)
        storeIndexToFile(temp)

    return 0





def process_feature(args):
        """
        Process a single feature to generate index data.
        """
        feature, i, key = args
        temp = {}
        feature_vec = feature

        # start_time = time.time()
        # logging.info(f"Task {i} started at {start_time}")

        E = indexGenQ(feature_vec, key)  # Generate the index using the provided function

        temp["id"] = i
        temp["E"] = E.tolist()
        storeIndexToFile(temp)
        # end_time = time.time()
        # logging.info(f"Task {i} ended at {end_time}")

def createIndex_MUTI(features, key):
    """
    Optimized createIndex function using multithreading.
    """

    tasks = [(feature, i, key) for i, feature in enumerate(features)]
    # Use ThreadPoolExecutor for multithreading
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        # Use tqdm to monitor progress
        for _ in tqdm(pool.imap_unordered(process_feature, tasks), total=len(tasks)):
            pass  # Wait for all processes to complete
    #  chunksize=1
    return 0

def storeIndexToFile(index_list, filename="index_query.jsonl"):

    # for index in tqdm(index_list):
    # for index in index_list:
    json_object = json.dumps(index_list)
    with open(filename, "a", encoding="Utf-8") as outfile:
        outfile.write(json_object)
        outfile.write('\n')

def euclidean_distance_np(a, b):
    """
    计算两个 NumPy 数组之间的欧式距离
    """
    diff = a - b
    squared_diff = diff**2
    sum_squared = np.sum(squared_diff)
    return np.sqrt(sum_squared)

if __name__ == "__main__":
    start = time.perf_counter()
    print("Reading Key from file ...")

    key = readKeyFromFile()
    print("Reading Features from file ...")
    gallery_fea, features_id = readFeatures("gallery")
    query_fea, features_id = readFeatures("query")
    query_fea_pac, gallery_fea_pca = DimEncfea(query_fea, gallery_fea)

    features = np.split(gallery_fea_pca, gallery_fea_pca.shape[0])
    # features = np.split(query_fea_pac, query_fea_pac.shape[0])
    rows = len(features)
    cols = features[0].shape[1]
    # print(f"Total images: , dimension : , rows, ")
    print("Total images:", rows)  # 输出: 数组的行数（长度）为: 3
    print("dimension:", cols)  # 输出: 数组的列数（宽度）为: 3
    # id = np.arange(0, rows)
    # fea = {'id': id, 'features_vec': features_vec}

    print("Creating index ...")
    index_list = createIndex(features, key)
    end = time.perf_counter()
    print(end - start)
    print("Storing index to file ...")
    # storeIndexToFile(index_list)


