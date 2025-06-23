import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from encrypt import chaotic_map_enc
from encrypt import get_image_block

class Mydata(Dataset):
    def __init__(self, root_dir, lable_dir):  # 魔术方法传入的是数据集的两个路径
        self.root_dir = root_dir  # 建立变量存储文件地址
        self.lable_dir = lable_dir  # 存储目录
        self.path = os.path.join(self.root_dir, self.lable_dir)  # 合成目录文件
        self.img_path = os.listdir(self.path)  # 存储目录文件到列表

    def __getitem__(self, idx):  # 字典方法
        img_name = self.img_path[idx]  # 将序号是IDX图片的位置存储
        img_item_path = os.path.join(self.root_dir, self.lable_dir, img_name)  # 得到每一个图片的位置
        img = Image.open(img_item_path)
        lable = self.lable_dir
        # print(1)
        return img, lable

    def __len__(self):
        return len(self.img_path)

def read_image(path):
    img = cv2.imread(path)
    return img

root_dir = r"E:\Github\PyRetri-master\data\CUB_200_2011\images" # 一级路径
lable_dir = "001.Black_footed_Albatross" # 二级路径
dataset = Mydata(root_dir,lable_dir) # 路径拼合
encdata = dataset # 路径存储
pix = encdata[0]
pix = np.array(pix[0])
# gary = cv2.cvtColor(pix, cv2.COLOR_BGR2GRAY)
enc_img = chaotic_map_enc(pix, "12131426241321231312")
# enc_img.cv2.save(r"E:\Github\PyRetri-master\chaotic_map_encryption\enc_data")
cv2.imwrite(r"E:\Github\PyRetri-master\chaotic_map_encryption\enc_data\jiami.jpeg", enc_img)