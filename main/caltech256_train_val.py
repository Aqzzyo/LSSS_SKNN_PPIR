import os

datasetpath = r"E:\Github\PyRetri-master\data\caltech256"

dirs = os.listdir(datasetpath)
dirs.sort()
with open(r'label.txt', 'w', encoding='utf-8') as f:
    for i in dirs:
        f.write(i)
        f.write('\n')

# print(dirs)
it = 0
Matrix = [[] for x in range(257)]  # all filenames under DATA_PATH
for d in dirs:
    for _, _, filename in os.walk(os.path.join(datasetpath, d)):
        for i in filename:
            Matrix[it].append(
                os.path.join(os.path.join(datasetpath, d), i))  # filename is a list of pic files under the fold
    it = it + 1

# print(Matrix)
with open(r'dataset-val.txt', 'w', encoding='utf-8') as f:
    for i in range(len(Matrix)):
        for j in range(10):
            f.write(os.path.join(datasetpath, Matrix[i][j]))
            f.write(' ')
            f.write(str(i))
            f.write('\n')
with open(r'dataset-trn.txt', 'w', encoding='utf-8') as f:
    for i in range(len(Matrix)):
        for j in range(10, len(Matrix[i])):
            f.write(os.path.join(datasetpath, Matrix[i][j]))
            f.write(' ')
            f.write(str(i))
            f.write('\n')
