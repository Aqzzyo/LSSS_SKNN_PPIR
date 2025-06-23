import os
import numpy as np
import pandas as pd

# label = pd.read_csv('../dataset_1/label_1.csv')
# label = np.array(label)
# label = label.tolist()
target = '0'


# for i in range(len(label)):
#     for j in range(len(label[i])):
#         target += str(label[i][j]) + ' '
#     print(target)
#     target = ''
def generate(dir):
    files = os.listdir(dir)  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    files_gallery = files[:75]
    files_query = files[76:77]
    # files.sort()  #对文件或文件夹进行排序
    # files.sort(key=lambda x: int(x.replace("frame", "").split('.')[0]))
    print('****************')
    print('input :', dir)
    print('start...')
    target = '0'
    target1 = '1'
    i = 0
    listText = open(r'E:\Github\PyRetri-master\main\split_file\75-1.txt', 'a+')  # 创建并打开一个txt文件，a+表示打开一个文件并追加内容
    # listText.truncate(0)  # 清空txt文件里的内容
    for file in files_gallery:  # 遍历文件夹中的文件
        fileType = os.path.split(file)  # os.path.split（）返回文件的路径和文件名，【0】为路径，【1】为文件名
        if fileType[1] == '.txt':  # 若文件名的后缀为txt,则继续遍历循环，否则退出循环
            continue
        name = outer_path + '/' + folder + '/' + file  # name 为文件路径和文件名+空格+label+换行
        name = name + ' ' + target + '\n'
        i += 1
        listText.write(name)  # 在创建的txt文件中写入name
    for file in files_query:  # 遍历文件夹中的文件
        fileType = os.path.split(file)  # os.path.split（）返回文件的路径和文件名，【0】为路径，【1】为文件名
        if fileType[1] == '.txt':  # 若文件名的后缀为txt,则继续遍历循环，否则退出循环
            continue
        name = outer_path + '/' + folder + '/' + file  # name 为文件路径和文件名+空格+label+换行
        name = name + ' ' + target1 + '\n'
        i += 1
        listText.write(name)  # 在创建的txt文件中写入name
    listText.close()  # 关闭txt文件
    print('down!')
    print('****************')


# outer_path = r'E:\Github\PyRetri-master\data\cifar-10-batches-py\train_file'  # 这里是你的图片路径
outer_path = r"E:\Github\PyRetri-master\data\caltech256"

if __name__ == '__main__':  # 主函数
    folderlist = os.listdir(outer_path)  # 列举文件夹

    for folder in folderlist:  # 遍历文件夹中的文件夹(若engagement文件夹中存在txt或py文件，则后面会报错）
        generate(os.path.join(outer_path, folder))  # 调用generate函数，函数中的参数为：（图片路径+文件夹名，标签号）