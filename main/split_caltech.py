import os
import shutil


# 创建训练集和验证集文件夹（如果不存在）
os.makedirs(r'E:\Github\PyRetri-master\data\caltech256\train', exist_ok=True)
os.makedirs(r'E:\Github\PyRetri-master\data\caltech256\val', exist_ok=True)

# 处理训练集文本文件
with open('dataset-trn.txt', 'r') as f_train:
    lines_train = f_train.readlines()
    for line in lines_train:
        parts = line.strip().split(' ')
        img_path = parts[0]
        label = parts[1]
        label_folder_train = os.path.join('train_set', label)
        os.makedirs(label_folder_train, exist_ok=True)
        shutil.copy(img_path, os.path.join(label_folder_train, os.path.basename(img_path)))

# 处理验证集文本文件
with open('dataset-val.txt', 'r') as f_val:
    lines_val = f_val.readlines()
    for line in lines_val:
        parts = line.strip().split(' ')
        img_path = parts[0]
        label = parts[1]
        label_folder_val = os.path.join('val_set', label)
        os.makedirs(label_folder_val, exist_ok=True)
        shutil.copy(img_path, os.path.join(label_folder_val, os.path.basename(img_path)))