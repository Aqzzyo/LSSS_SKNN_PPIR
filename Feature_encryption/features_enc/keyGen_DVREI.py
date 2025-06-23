# for imports as runtime
import sys
import json
import numpy as np
import time

sys.path.append("..")
import key_gen as KG

NP_KEYS = ["M", "M_inv"]
import random
import numpy as np

import random


def generate_random_integer(min_val=0, max_val=100, count=1):
    """
    生成随机整数

    参数:
    min_val -- 最小值（包含该值，默认为0）
    max_val -- 最大值（包含该值，默认为100）
    count -- 要生成的整数数量（默认为1）

    返回:
    单个整数（count=1时）或包含随机整数的列表（count>1时）

    异常:
    ValueError -- 如果参数无效
    """
    # 验证参数有效性
    if not isinstance(min_val, int) or not isinstance(max_val, int):
        raise TypeError("最小值和最大值必须是整数")

    if min_val > max_val:
        raise ValueError("最小值不能大于最大值")

    if not isinstance(count, int) or count < 1:
        raise ValueError("数量必须是正整数")

    # 处理特殊情况（范围大小为1）
    if min_val == max_val:
        return min_val if count == 1 else [min_val] * count

    # 生成随机整数
    if count == 1:
        return random.randint(min_val, max_val)
    else:
        return [random.randint(min_val, max_val) for _ in range(count)]





def keyGen(key_size, dimension):
    kmg = KG.generate_hex_key(key_size)
    M, M_inv = KG.generate_random_non_singular_matrix(dimension)

    lam = generate_random_integer()
    # create result dictionary
    result = {}
    result["M"] = M
    result["M_inv"] = M_inv
    result["lamda"] = lam


    return result


def storeKeyToFile(key):
    for dict_key in NP_KEYS:
        key[dict_key] = key[dict_key].tolist()

    json_object = json.dumps(key)
    with open("key_DVREI.json", "w") as outfile:
        outfile.write(json_object)
        outfile.write('\n')


def readKeyFromFile(filename="key_DVREI.json"):
    with open(filename, 'r') as fp:
        for line in fp:
            key = json.loads(line)

    for dict_key in NP_KEYS:
        key[dict_key] = np.array(key[dict_key], dtype='float')

    return key


if __name__ == "__main__":
    start = time.perf_counter()

    # system args
    # keysize = int(sys.argv[1])
    # dimension = int(sys.argv[2])
    keysize = int(80)
    dimension = int(1019)
    print("Generating Key ... ")
    key = keyGen(keysize, dimension)
    end = time.perf_counter()
    print(end - start)
    print("Storing key to file key.json ... ")
    storeKeyToFile(key)