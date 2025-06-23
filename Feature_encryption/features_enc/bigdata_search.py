import json
import numpy as np
import torch
import time
from tqdm import tqdm
import orjson  # 仅用于高性能解析


def stream_jsonl(file_path):
    """
    分块流式读取 JSONL 文件（每次处理 chunk_size 行）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk = []
        for line in f:
            line = line.strip()
            if line:
                chunk.append(line)
                if len(chunk) >= 1000:  # 每次处理 1000 行
                    yield chunk
                    chunk = []
        if chunk:  # 处理剩余数据
            yield chunk


def extract_dicts_from_json(file_path):
    """
    逐行解析 JSONL 文件（避免一次性加载全部数据）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    # 修复：仅传递一个参数（JSON 字符串）
                    yield orjson.loads(line)
                except orjson.JSONDecodeError as e:
                    print(f"JSON 解析错误: {e}")
                    continue


def transform_features(data_generator, max_features=12850):
    """
    分批转换特征数据（避免一次性存储全部数据）
    """
    features = []
    with tqdm(data_generator, desc="Processing Features", unit="item") as pbar:
        for idx, item in enumerate(pbar):
            if idx >= max_features:
                break
            try:
                # 确保字段存在且类型正确
                feature = {
                    'id': item['id'],
                    'E': np.array(item['E'], dtype=np.float64)  # 使用 float32 减少内存
                }
                features.append(feature)
            except KeyError as e:
                print(f"字段缺失: {e}")
        return features

def transform_features_query(data_generator, max_features=3855):
    """
    分批转换特征数据（避免一次性存储全部数据）
    """
    features = []
    with tqdm(data_generator, desc="Processing query Features", unit="item") as pbar:
        for idx, item in enumerate(pbar):
            if idx >= max_features:
                break
            try:
                # 确保字段存在且类型正确
                feature = {
                    'id': item['id'],
                    'Q': np.array(item['Q'], dtype=np.float64)  # 使用 float32 减少内存
                }
                features.append(feature)
            except KeyError as e:
                print(f"字段缺失: {e}")
        return features

def cal_distance(query_fea, gallery_fea):
    """优化矩阵乘法维度匹配"""
    inner_dot = torch.mm(gallery_fea, query_fea)
    return torch.trace(inner_dot)


if __name__ == "__main__":
    # 配置参数
    GALLERY_CHUNK_SIZE = 1000  # 图库分块大小
    QUERY_BATCH_SIZE = 15  # 查询批次大小
    MAX_FEATURES = 19275  # 图库最大特征数
    MAX_FEATURES_QUERY = 257
    file_gallery_path = r'E:\Github\PyRetri-master\data\features\features_enc\encrypted_features\precision\d128n20000GeM\index_gallery.jsonl'  # 替换为您的 JSON 文件路径
    file_query_path = r'E:\Github\PyRetri-master\data\features\features_enc\encrypted_features\precision\d128n20000GeM\index_query.jsonl'

    # 分块读取图库数据
    gallery_gen = extract_dicts_from_json(file_gallery_path)
    gallery_features = transform_features(gallery_gen, MAX_FEATURES)

    # 分批处理查询数据
    query_gen = extract_dicts_from_json(file_query_path)
    query_features = transform_features_query(query_gen, MAX_FEATURES_QUERY)


    total_correct = 0
    total_samples = 0
    pry = []
    time_start = time.time()

    for q in tqdm(range(0, MAX_FEATURES_QUERY), desc='Processing'):
        time_start1 = time.time()
        count = 0
        dis_sort = []
        for i in range(0, MAX_FEATURES):
            temp = {}
            query_fea = torch.from_numpy(query_features[q]['Q'])
            gallery_fea = torch.from_numpy(gallery_features[i]['E'])
            distance = cal_distance(query_fea, gallery_fea)
            if distance >= 0:
                temp["id"] = i
                temp["distance"] = distance
                dis_sort.append(temp)
        dis_sort = sorted(dis_sort, key=lambda x: x["distance"], reverse=True)
        for item in dis_sort[:75]:
            if q*75 <= item["id"] < (q+1) * 75:
                count += 1
        pry.append(round(count / 75, 3))
        time_end1 = time.time()
        time_sum1 = time_end1 - time_start1  # 计算的时间差为程序的执行时间，单位为秒/s
        print(time_sum1)
            # print(i)

    total = sum(pry)
    # 计算平均值，通过总和除以元素个数
    average = total / len(pry)
    print(average)
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)
