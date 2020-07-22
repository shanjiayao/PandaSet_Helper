import os
import numpy as np
import math
import sys
from pandaset.dataset import DataSet


def split_dataset(dataset, split_rule, path):
    resule_file_path = path
    result_file_name = 'dataset_split.txt'
    result_file = os.path.join(resule_file_path, result_file_name)
    fp = open(result_file, 'w')
    sequence = dataset.sequences(with_semseg=True)
    seq_list = []
    for num, seq_num_str in enumerate(sequence):
        seq_list.append(seq_num_str)
    _len = len(seq_list)
    train_seq = seq_list[:int(_len * split_rule[0])]
    for i, num in enumerate(train_seq):
        fp.write("train" + " " + num + "\r\n")
    valid_seq = list(set(seq_list) - set(train_seq))
    valid_seq = valid_seq[:int(_len * split_rule[1])]
    for i, num in enumerate(valid_seq):
        fp.write("valid" + " " + num + "\r\n")
    test_seq = list(set(seq_list) - set(train_seq) - set(valid_seq))
    for i, num in enumerate(test_seq):
        fp.write("test" + " " + num + "\r\n")
    fp.close()


def get_mode_seqs_list(path, mode):
    file = path + "/" + "dataset_split" + ".txt"
    _list = []
    for line in open(file):
        str_line = line.strip().split()
        if str_line[0] == mode:
            _list.append(str_line[1])
    # print(_list)
    return _list


def get_name_by_read_dir(path):
    indexs = []
    for every_file in os.listdir(path):
        index = every_file.split('.')
        if len(index) is 2:
            indexs.append(index)
    indexs.sort()
    count = 0
    str_count = ''
    if len(indexs) == 0:
        str_count = '000000'
        # print("None")
    else:
        for i in range(len(indexs)):
            count += 1
            if count < 10:
                str_count = '00000' + str(count)
            elif count < 100:
                str_count = '0000' + str(count)
            elif count < 1000:
                str_count = '000' + str(count)
            elif count < 10000:
                str_count = '00' + str(count)
            elif count < 100000:
                str_count = '0' + str(count)
            # print('str_count: ', str_count)
    return str_count


def points2pcd(points, path, name):
    # 存放路径
    PCD_FILE_PATH = os.path.join(path, name)
    if os.path.exists(PCD_FILE_PATH):
        os.remove(PCD_FILE_PATH)

    # 写文件句柄
    handle = open(PCD_FILE_PATH, 'a')

    # 得到点云点数
    point_num = points.shape[0]

    # pcd头部（重要）
    handle.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')

    # 依次写入点
    for i in range(point_num):
        string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2])
        handle.write(string)
    handle.close()


def seg_label_save_txt(labels, path, name):
    label_file = os.path.join(path, name)
    fp = open(label_file, 'w')
    labels = np.squeeze(labels)
    labels = labels.tolist()
    for i in range(len(labels)):
        labels[i] = str(labels[i])
        fp.write(labels[i] + " ")
    fp.close()


def cub_label_save_txt(labels, path, name):
    label_file = os.path.join(path, name)
    fp = open(label_file, 'w')
    # print(type(labels[0]))
    for i in range(labels.shape[0]):
        # labels[i] = labels[i].tostring()
        for j in range(labels.shape[1]):
            label = labels[i][j]
            if isinstance(label, str) is not True:
                label = str(label)
            if j < labels.shape[1] - 1:
                fp.write(label + " ")
            elif j == labels.shape[1] - 1:
                fp.write(label + "\r\n")
    fp.close()


if __name__ == '__main__':
    path = '/media/echo/仓库卷/DataSet/Autonomous Driving/PandaSets/pandaset'
    datatest = DataSet(path)
    split_rule = [0.6, 0.2, 0.2]    # train/valid/test
    split_dataset(datatest, split_rule=split_rule, path=path)
    train_seqs = get_mode_seqs_list(path=path, mode='train')
    print(train_seqs)

    lidar_path = '/media/echo/仓库卷/DataSet/Autonomous Driving/PandaSets/my_pandaset/train/lidar'
    seg_label_path = '/media/echo/仓库卷/DataSet/Autonomous Driving/PandaSets/my_pandaset/train/label/semseg'
    cub_label_path = '/media/echo/仓库卷/DataSet/Autonomous Driving/PandaSets/my_pandaset/train/label/cuboids'

    seq = datatest[train_seqs[44]]
    print(train_seqs[44])
    seq.load()
    for i in range(len(seq.lidar.data)):
        # get name string
        start_str = get_name_by_read_dir(lidar_path)
        # point save as pcd
        points3d_lidar = seq.lidar.data[i].to_numpy()
        points3d_lidar_xyz = points3d_lidar[:, :3]
        # print('points3d_lidar_xyz', points3d_lidar_xyz.shape)
        points2pcd(points3d_lidar_xyz, lidar_path, start_str + '.pcd')
        # label save as txt
        seg_label = seq.semseg
        seg_label_per_frame = seg_label[i].to_numpy()
        # print('seg_label_per_frame', seg_label_per_frame.shape)
        seg_label_save_txt(seg_label_per_frame, seg_label_path, start_str + '.txt')

        cuboid_label = seq.cuboids
        cuboid_label_per_frame = cuboid_label[i].to_numpy()
        cub_label_save_txt(cuboid_label_per_frame, cub_label_path, start_str + '.txt')
        # print(cuboid_label_per_frame.shape)




