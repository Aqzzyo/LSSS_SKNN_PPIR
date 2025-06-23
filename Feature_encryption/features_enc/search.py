import json

import numpy as np
import torch
import time
from tqdm import tqdm
import orjson
def extract_dicts_from_json(file_path):
    dicts = []
    with open(file_path, 'r') as file:
        for line in tqdm(file):
            json_data = json.loads(line)
            if isinstance(json_data, dict):
                dicts.append(json_data)
    return dicts

def transform_narry(dicts):
    dicts_gallery = []
    for i in range(0, 12850):
        for key, value in dicts[i].items():

            if key == 'id':
                extracted_dicts_narrdy = {}
                extracted_dicts_narrdy[key] = value
            else:
                extracted_dicts_narrdy[key] = np.array(value)
        dicts_gallery.append(extracted_dicts_narrdy)

    return dicts_gallery

def transform_narry_query(dicts):
    dicts_gallery = []
    for i in range(0, 3855):
        for key, value in dicts[i].items():

            if key == 'id':
                extracted_dicts_narrdy = {}
                extracted_dicts_narrdy[key] = value
            else:
                extracted_dicts_narrdy[key] = np.array(value)
        dicts_gallery.append(extracted_dicts_narrdy)

    return dicts_gallery

def cal_distance(query_fea: torch.tensor, gallery_fea: torch.tensor) -> torch.tensor:
        """
        Calculate the distance between query set features and gallery set features.
        Args:
            query_fea (torch.tensor): query set features.
            gallery_fea (torch.tensor): gallery set features.
        Returns:
            dis (torch.tensor): the distance between query set features and gallery set features.
        """
        # query_fea = query_fea.transpose(1, 0)
        inner_dot = gallery_fea.mm(query_fea)
        dis = torch.trace(inner_dot)
        # dis = (gallery_fea ** 2).sum(dim=1, keepdim=True) + (query_fea ** 2).sum(dim=0, keepdim=True)
        # dis = dis - 2 * inner_dot
        # dis = dis.transpose(1, 0)
        return dis


if __name__ == "__main__":
    file_gallery_path = r'E:\Github\PyRetri-master\data\features\features_enc\encrypted_features\precision\d128GAP\index_gallery.jsonl'  # 替换为您的 JSON 文件路径
    file_query_path = r'E:\Github\PyRetri-master\data\features\features_enc\encrypted_features\precision\d128GAP\index_query.jsonl'
    extracted_dicts_gallery = transform_narry(extract_dicts_from_json(file_gallery_path))  # 从json文件中提取索引列表
    extracted_dicts_query = transform_narry_query(extract_dicts_from_json(file_query_path))
    dis_sort = []
    time_start = time.time()  # 记录开始时间
    pry = []
    for q in tqdm(range(0, 3855, 15), desc='Processing'):
        count = 0
        dis_sort = []
        for i in range(0, 12850):
            temp = {}
            query_fea = torch.from_numpy(extracted_dicts_query[q]['Q'])
            gallery_fea = torch.from_numpy(extracted_dicts_gallery[i]['E'])
            distance = cal_distance(query_fea, gallery_fea)
            if distance >= 0:
                temp["id"] = i
                temp["distance"] = distance
                dis_sort.append(temp)
        dis_sort = sorted(dis_sort, key=lambda x: x["distance"], reverse=True)
        for item in dis_sort[:5]:
            if (q//15)*50 <= item["id"] < (q//15+1) * 50:
                count += 1
        pry.append(round(count / 5, 3))
            # print(i)

    total = sum(pry)
    # 计算平均值，通过总和除以元素个数
    average = total / len(pry)
    print(average)
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)







