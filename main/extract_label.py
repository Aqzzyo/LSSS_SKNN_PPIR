import os

datasetpath = r"E:\Github\PyRetri-master\data\256_ObjectCategories"

d = os.listdir(datasetpath)
d.sort()
with open(r'label.txt', 'w', encoding='utf-8') as f:
    for i in d:
        f.write(i)
        f.write('\n')
