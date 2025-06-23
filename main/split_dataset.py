# -*- coding: utf-8 -*-
import argparse
import os

from pyretri.extract.utils import split_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--dataset', '-d', default=r'E:\Github\PyRetri-master\data\caltech256', type=str, help="path for the dataset.")
    parser.add_argument('--split_file', '-sf', default=r'E:\Github\PyRetri-master\main\split_file\75-1.txt', type=str, help="name for the dataset.")

    args = parser.parse_args()

    return args


def main():

    # init args
    args = parse_args()
    assert args.dataset is not None, 'the dataset must be provided!'
    assert args.split_file is not None, 'the save path must be provided!'

    # split dataset
    split_dataset(args.dataset, args.split_file)

    print('split dataset have done!')


if __name__ == '__main__':
    main()
