import random


def generate_keyb(num):
    lst = list(range(num))
    result = lst.copy()
    n = len(result)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        result[i], result[j] = result[j], result[i]
    return result

def generate_keys(num):
    lst = list(range(num))
    result = lst.copy()
    n = len(result)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        result[i], result[j] = result[j], result[i]
    return result

def generate_keyp(num):
    lst = list(range(num))
    result = lst.copy()
    n = len(result)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        result[i], result[j] = result[j], result[i]
    return result

def generate_keyv(num):
    result = []
    for _ in range(num):
        num = random.randint(0, 3)
        result.append(num)
    return result


