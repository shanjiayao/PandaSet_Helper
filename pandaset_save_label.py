"""
/** @file pandaset_save_label.py
 *  @version 1.0
 *  @date 2020.6.23
 *
 *  @brief 本文件是作为数据转换工具使用的，大概的流程是：加载数据集和对应的序列，然后程序会将对应标签的点云存成pcd文件以及txt标签文件
 *
 *  1. 对数据集中每一个序列每一帧的点云中的对应的标签读取bbox位置
 *  2. 将在框中的点云存成pcd文件，且pcd中点的第四个维度为语义的类别标签
 *  3. 在bbox的label中，把当前序列中的帧序号加入到第一个位置。

    semseg_label:
        {'1': 'Smoke',
        '2': 'Exhaust',
        '3': 'Spray or rain',
        '4': 'Reflection',
        '5': 'Vegetation',
        '6': 'Ground',
        '7': 'Road',
        '8': 'Lane Line Marking',
        '9': 'Stop Line Marking',
        '10': 'Other Road Marking',
        '11': 'Sidewalk',
        '12': 'Driveway',
        '13': 'Car',
        '14': 'Pickup Truck',
        '15': 'Medium-sized Truck',
        '16': 'Semi-truck',
        '17': 'Towed Object',
        '18': 'Motorcycle',
        '19': 'Other Vehicle - Construction Vehicle',
        '20': 'Other Vehicle - Uncommon',
        '21': 'Other Vehicle - Pedicab',
        '22': 'Emergency Vehicle',
        '23': 'Bus',
        '24': 'Personal Mobility Device',
        '25': 'Motorized Scooter',
        '26': 'Bicycle',
        '27': 'Train',
        '28': 'Trolley',
        '29': 'Tram / Subway',
        '30': 'Pedestrian',
        '31': 'Pedestrian with Object',
        '32': 'Animals - Bird',
        '33': 'Animals - Other',
        '34': 'Pylons',
        '35': 'Road Barriers',
        '36': 'Signs',
        '37': 'Cones',
        '38': 'Construction Signs',
        '39': 'Temporary Construction Barriers',
        '40': 'Rolling Containers',
        '41': 'Building',
        '42': 'Other Static Object'}
 *  @author shan
 *
 */
"""

import numpy as np
import os

# pandaset
from pandaset.dataset import DataSet
from pandaset.geometry import center_box_to_corners
import shutil


def split_dataset(dataset, split_rule, path):
    resule_file_path = path
    result_file_name = 'dataset_split.txt'
    result_file = os.path.join(resule_file_path, result_file_name)
    with open(result_file, 'w') as fp:
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


def get_mode_seqs_list(path, mode):
    file = path + "/" + "dataset_split" + ".txt"
    _list = []
    for line in open(file):
        str_line = line.strip().split()
        if str_line[0] == mode:
            _list.append(str_line[1])
    # print(_list)
    return _list


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
        string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2]) + ' ' + str(points[i, 3])
        handle.write(string)
    handle.close()


def make_dir_valid(dir):
    if os.path.exists(dir) is not True:
        os.mkdir(dir)


def empty_dir(dir):
    shutil.rmtree(dir)
    os.mkdir(dir)


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


def seg_label_save_txt(labels, path, name):
    """

    :param labels: labels can not be single value like 13, must be a list
    :param path:
    :param name:
    :return:
    """
    label_file = os.path.join(path, name)
    with open(label_file, 'w') as fp:
        labels = np.squeeze(labels)
        labels = labels.tolist()
        for i in range(len(labels)):
            labels[i] = str(labels[i])
            fp.write(labels[i] + " ")


def box_label_save_txt(labels, path, name):
    label_file = os.path.join(path, name)
    with open(label_file, 'w') as fp:
        length = labels.shape[0]    # length is 17
        for i in range(length):
            label = labels[i]
            if isinstance(label, str) is not True:
                label = str(label)
            if i < length - 1:
                fp.write(label + " ")
            elif i == length - 1:
                fp.write(label + "\r\n")


if __name__ == '__main__':
    # Path
    dataset_original_path = '/media/echo/仓库卷/DataSet/Autonomous Driving/PandaSets/pandaset'
    pcd_original_path = '/media/echo/仓库卷/DataSet/Autonomous Driving/PandaSets/Dataset/train/lidar'
    seg_original_path = '/media/echo/仓库卷/DataSet/Autonomous Driving/PandaSets/Dataset/train/labels/semsegs'
    box_original_path = '/media/echo/仓库卷/DataSet/Autonomous Driving/PandaSets/Dataset/train/labels/cuboids'
    # empty_dir(pcd_original_path)
    # empty_dir(seg_original_path)
    # empty_dir(box_original_path)
    # exit()

    # Create DataSet object
    dataset = DataSet(dataset_original_path)

    # Set split rules
    split_rule = [0.8, 0.1, 0.1]    # train/valid/test

    # Generate 'dataset_split.txt'
    split_dataset(dataset, split_rule=split_rule, path=dataset_original_path)

    # Get the sequences numbers of train dataset
    train_seqs = get_mode_seqs_list(path=dataset_original_path, mode='train')
    print(train_seqs)

    # Init the special Seq with seq_num
    seq = dataset[train_seqs[5]] # 5 is over
    print(train_seqs[5])
    seq.load()

    for frame_num in range(len(seq.lidar.data)):    # len(seq.lidar.data) is 80
        points3d_lidar = seq.lidar.data[frame_num].to_numpy()
        points3d_lidar_xyz = points3d_lidar[:, :3]  # 取前三位，点云的xyz坐标

        box_label = seq.cuboids.data[frame_num].to_numpy()
        seg_label = seq.semseg.data[frame_num].to_numpy()
        # print(box_label.shape)

        # 取一帧点云，读每一行label
        for label_row in range(box_label.shape[0]):
            # if box_label[i][1] == 'Car':
            box = []
            # print(box_label[i][1:11])
            box = box_label[label_row][5:11]
            box = box.tolist()
            box.append(box_label[label_row][2])
            corners = center_box_to_corners(box)

            x_min = min(corners[:, 0])
            x_max = max(corners[:, 0])
            y_min = min(corners[:, 1])
            y_max = max(corners[:, 1])
            z_min = min(corners[:, 2])
            z_max = max(corners[:, 2])

            pcd_points = []
            semseg_points_label = []
            now_bbox_label = []
            points_less_flag = False
            for points_num in range(points3d_lidar_xyz.shape[0]):
                if (points3d_lidar_xyz[points_num][0] > x_min) \
                        and (points3d_lidar_xyz[points_num][0] < x_max) \
                        and (points3d_lidar_xyz[points_num][1] > y_min) \
                        and (points3d_lidar_xyz[points_num][1] < y_max) \
                        and (points3d_lidar_xyz[points_num][2] > z_min) \
                        and (points3d_lidar_xyz[points_num][2] < z_max):
                    pcd_points.append(points3d_lidar_xyz[points_num])
                    semseg_points_label.append(seg_label[points_num])

            # path
            pcd_path = pcd_original_path
            seg_path = seg_original_path
            box_path = box_original_path
            pcd_path = os.path.join(pcd_path, box_label[label_row][1])
            seg_path = os.path.join(seg_path, box_label[label_row][1])
            box_path = os.path.join(box_path, box_label[label_row][1])
            make_dir_valid(pcd_path)
            make_dir_valid(seg_path)
            make_dir_valid(box_path)

            # name
            if len(pcd_points) <= 1:
                continue
            if len(pcd_points) < 10:
                pcd_path = os.path.join(pcd_path, 'less')
                seg_path = os.path.join(seg_path, 'less')
                box_path = os.path.join(box_path, 'less')
                make_dir_valid(pcd_path)
                make_dir_valid(seg_path)
                make_dir_valid(box_path)
            # print(pcd_path)
            name_str = get_name_by_read_dir(pcd_path)
            pcd_name = name_str + '.pcd'
            label_name = name_str + '.txt'

            # save points 4d with xyz and seg label
            pcd_points = np.array(pcd_points)
            semseg_points_label = np.squeeze(np.array(semseg_points_label))  # np.array(semseg_points_label) is (184, 1) => (184)
            points_4d = np.zeros([pcd_points.shape[0], 4])
            points_4d[:, :3] = pcd_points
            points_4d[:, 3] = semseg_points_label.astype(np.float32)
            points2pcd(points_4d, pcd_path, pcd_name)

            # save seg label as .txt file
            seg_label_save_txt(semseg_points_label, seg_path, label_name)

            # save box label as .txt file
            current_row_label = box_label[label_row]    # np.array()
            current_row_label = np.insert(current_row_label, 0, frame_num)
            box_label_save_txt(current_row_label, box_path, label_name)
            print(f"Running to save pcd and labels.....percentage is {frame_num/0.8}%")
