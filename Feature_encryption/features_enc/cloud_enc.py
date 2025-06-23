import numpy as np
import sys
import multiprocessing
sys.path.append('/root/autodl-tmp/pyretri/')
from pyretri.index.utils.feature_loader import FeatureLoader
import random
from tqdm import tqdm
from index_gen import index_gen as IG
import general_utils as util
import json
from keyGen import readKeyFromFile
from cloud_dim import DimEncfea
import time
import concurrent.futures

featureLoader = FeatureLoader()


def readFeatures(filename):
    query_fea_dir = f"/root/autodl-tmp/pyretri/data/256_ObjectCategories/{filename}"
    gallery_fea_dir = f"/root/autodl-tmp/pyretri/data/256_ObjectCategories/{filename}"
    feature_names = ["pool5_GAP"]

    if filename == "query":
        query_fea, query_info, _ = featureLoader.load(query_fea_dir, feature_names)
        return query_fea, query_info
    else:
        gallery_fea, gallery_info, _ = featureLoader.load(gallery_fea_dir, feature_names)
        return gallery_fea, gallery_info


def indexGen(feature_vec, key):
    dimension = feature_vec.shape[1]
    M1 = key["M1"]
    M2 = key["M2"]
    permutation = key["permutation"]
    # choose randm num between 1-100.
    # alpha = random.randint(1, 100)
    alpha = 1
    rf = random.randint(1, 100)

    ext_feature_vec = IG.extend_feature_vector(feature_vec, alpha, rf)
    ext_feature_vec = np.array(ext_feature_vec)
    permuted_fv = util.permute_feature_vector(ext_feature_vec, permutation)

    F = util.create_diagonal_matrix(permuted_fv)
    S = util.generate_lower_triangular_matrix(dimension)
    E = util.multiply_matrices([M1, S, F, M2])

    return E


def indexGenQ(feature_vec, key):
    dimension = feature_vec.shape[1]
    M1_inv = key["M1_inv"]
    M2_inv = key["M2_inv"]
    permutation = key["permutation"]
    # choose randm num between 1-100.
    # beta = random.randint(1, 100)
    beta = 1
    rq = random.randint(1, 100)
    sim = 0.5
    c = 1
    theta = IG.convert_sim_to_theta(sim, c)

    ext_feature_vec = IG.extend_feature_vector_Q(feature_vec, beta, rq, theta)
    ext_feature_vec = np.array(ext_feature_vec)
    permuted_fv = util.permute_feature_vector(ext_feature_vec, permutation)

    Q = util.create_diagonal_matrix(permuted_fv)
    S = util.generate_lower_triangular_matrix(dimension)

    D = util.multiply_matrices([M2_inv, Q, S, M1_inv])

    return D


# def createIndex(features, key):
#
#     results = []
#     i = 0
#
#     for feature in tqdm(features):
#         temp = {}
#         id = i
#         feature_vec = feature
#
#         E = indexGen(feature_vec, key)
#
#         temp["id"] = id
#         temp["E"] = E.tolist()
#         i = i+1
#         # storeIndexToFile(temp)
#         # results.append(temp)
#
#     return 0
def process_feature(args):
        """
        Process a single feature to generate index data.
        """
        feature, i, key = args
        temp = {}
        feature_vec = feature
        E = indexGen(feature_vec, key)  # Generate the index using the provided function

        temp["id"] = i
        temp["E"] = E.tolist()
        storeIndexToFile(temp)


def createIndex(features, key):
    """
    Optimized createIndex function using multithreading.
    """

    tasks = [(feature, i, key) for i, feature in enumerate(features)]
    # Use ThreadPoolExecutor for multithreading
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        # Use tqdm to monitor progress
        for _ in tqdm(pool.imap_unordered(process_feature, tasks), total=len(tasks)):
            pass  # Wait for all processes to complete

    return 0


def storeIndexToFile(index_list, filename="index.jsonl"):
    for index in tqdm(index_list):
        json_object = json.dumps(index)
        with open(filename, "a") as outfile:
            outfile.write(json_object)
            outfile.write('\n')


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
