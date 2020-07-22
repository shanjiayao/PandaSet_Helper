import torch.utils.data as torch_data
import os
import random
import numpy as np
import sys
from typing import List, Dict, Tuple
sys.path.append('..')
from utils.config import cfg
import utils.box_util as box_util
from pandaset.geometry import center_box_to_corners


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


def readPCD(path):
    """Read .pcd file and get points

    :param path: path of .pcd file with type of [string]
    :return: points with type of [numpy array]
    """
    with open(path, 'r') as f:
        count = 1
        for line in f:
            if count > 11:
                ls = line.strip().split()
                # only use XYZ
                point = np.append(float(ls[0]), float(ls[1]))
                point = np.append(point, float(ls[2]))
                point = np.append(point, float(ls[3]))
                if count == 12:
                    points = point
                else:
                    points = np.row_stack((points, point))
            else:
                pass
                # print(line)
            count += 1
    return points


class PandaDataset(torch_data.Dataset):
    """Pytorch subclass of torch_data.Dataset to load PandaSet.

        Args:
             pandaset_path: Absolute or relative path where PandaSet has been extracted to.
             mode: Mode of Network
             classes:
                    '1': 'Smoke',
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
                    '42': 'Other Static Object'

        Attributes:

        """
    def __init__(self, root_dir, npoints, mode, classes) -> None:
        self._mode = mode.lower()
        self._class = classes
        self._sample_points_num = npoints
        self._index_step = 20
        self._resample_pts_threshold = 0.9
        self._pcd_dir = os.path.join(root_dir, self._mode, 'lidar', self._class)
        self._cuboids_dir= os.path.join(root_dir, self._mode, 'labels', 'cuboids', self._class)
        self._semseg_dir = os.path.join(root_dir, self._mode, 'labels', 'semseg', self._class)
        self._dataset_length = len([x for x in os.listdir(self._pcd_dir) if not os.path.isdir(x)])
        print('Dataset includes {0} pcd files\n'.format(self._dataset_length))

    def __getitem__(self, index) -> Dict:
        # 根据传入的参数，确定当前的pcd文件夹下，搜索点云和模板点云的序号
        search_index, template_index = self.get_pair_index(index)

        # 通过读取pcd文件，得到搜索点云和模板点云，注意这里的点云维度是4，　最后一个维度是语义信息
        search_points = self.get_pts_from_pcd(search_index)
        template_points = self.get_pts_from_pcd(template_index)

        # 从label的txt文件中读取对应的bbox信息，这里与kitti不同的是，kitti_rpn_dataset.py中读取的bbox信息，xyz是bbox底部的中心点坐标
        # 而这里是整个bbox的中心点坐标
        search_box = self.get_bbox_from_label(search_index)
        template_box = self.get_bbox_from_label(template_index)

        # 通过模板点云和模板点云的真值bbox两部分，来对模板点云进行操作，过程是扩大bbox，然后使用in_hull函数判断，生成符合条件的模板点云
        # 这里的输入必须是三维的点云，所以得到的也是三维点云，第四个语义维度被舍弃
        # TODO: in_hull函数还没理解作用...
        template_points_3d = self.template_point_prepare(template_points[:, 0:3], template_box)

        # 重采样搜索点云和模板点云，　使数量均为500
        # 这里使用的策略是，点云数量大于500，则根据阈值划分远近，各自重采样后再加和
        # 如果点云数量小于500，　则不够的部分由已有点云随机采样获得，最后加和得到500点
        # 这里的搜索点云为了生成最后的语义标签，仍然是四维度的点云矩阵
        search_points = self.resample_pts(search_points)
        template_points_3d = self.resample_pts(template_points_3d)

        # 点云数据增广，根据配置文件中的增广方法进行点云的数据扩充
        aug_search_points = search_points.copy()
        aug_search_box = search_box.copy()
        if cfg.AUG_DATA and self._mode == 'train':
            aug_search_points, aug_search_box = self.data_augmentation(aug_search_points, aug_search_box)

        # 生成伪labels
        cls_label, reg_label, seg_label = self.generate_rpn_training_label(aug_search_points, aug_search_box)

        # 存入字典
        data = {'template_data': template_points[0, 0:3], 'search_data': aug_search_points,
                'cls_label': cls_label, 'reg_label': reg_label, 'seg_label': seg_label,
                'true_box': aug_search_box}
        return data

    def __len__(self) -> int:
        return self._dataset_length

    def get_pair_index(self, index) -> Tuple[int, int]:
        index_1 = index
        min_idx = max(0, (index_1 - self._index_step))
        max_idx = min(self._dataset_length - 1, (index_1 + self._index_step))
        index_2 = random.randint(min_idx, max_idx)
        return index_1, index_2

    def get_pts_from_pcd(self, index) -> np.ndarray:
        pcd_file_path = os.path.join(self._pcd_dir, '%06d.pcd' % index)
        print('Pcd file path is {0}\n'.format(pcd_file_path))
        assert os.path.exists(pcd_file_path)   # if the path is exist or not
        lidar_data = readPCD(pcd_file_path)  # with shape of (N, 4)
        return lidar_data

    def resample_pts(self, pts) -> np.ndarray:
        # if points num is more than 500, select the near pts and far pts, then concatenate them
        print('Input pts num is {0}'.format(pts.shape))
        if self._sample_points_num < len(pts):
            pts_depth = abs(pts[:, 2])
            pts_near_flag = pts_depth < self._resample_pts_threshold

            far_idxs = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]

            far_sample_npoints = int(float(len(far_idxs)) / float(len(pts)) * self._sample_points_num)
            near_sample_npoints = self._sample_points_num - far_sample_npoints

            near_idxs_choice = np.random.choice(near_idxs, near_sample_npoints, replace=False)
            far_idxs_choice = np.random.choice(far_idxs, far_sample_npoints, replace=False)
            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0)
            np.random.shuffle(choice)
        # if points num is less than 500, use random choice to resample upto 500
        else:
            choice = np.arange(0, len(pts), dtype=np.int32)
            if self._sample_points_num > len(pts):
                extra_choice = np.random.choice(choice, self._sample_points_num - len(pts), replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        new_points = pts[choice, :]
        print('Output pts num is {0}\r\n'.format(new_points.shape))
        return new_points

    def get_bbox_from_label(self, index) -> np.ndarray:
        box_file_path = os.path.join(self._cuboids_dir, '%06d.txt' % index)
        print('Box file path is {0}\n'.format(box_file_path))
        assert os.path.exists(box_file_path)
        bbox = np.zeros(7)
        with open(box_file_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                _temp = np.array(line[6:12]).astype(np.float)  # List[string] -> np.array -> np.float
                bbox[0:6] = _temp  # x, y, z, w, h, l    xyz are the center point coordination
                bbox[6] = float(line[3])   # yaw [radians]
        return bbox

    def data_augmentation(self, aug_pts_rect, aug_gt_boxes3d, mustaug=False, stage=1) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param aug_pts_rect: (N, 3)
        :param aug_gt_boxes3d: (N, 7)
        :param mustaug: bool
        :param stage: int
        :return:
        """
        # 增加一个维度，沿着0方向
        aug_gt_boxes3d = np.expand_dims(aug_gt_boxes3d, axis=0)

        # 获取点云增广的方法列表，包括旋转、缩放、切片
        aug_list = cfg.AUG_METHOD_LIST
        aug_enable = 1 - np.random.rand(3)
        if mustaug is True:
            aug_enable[0] = -1
            aug_enable[1] = -1

        if 'rotation' in aug_list and aug_enable[0] < cfg.AUG_METHOD_PROB[0]:
            angle = np.random.uniform(-np.pi / cfg.AUG_ROT_RANGE, np.pi / cfg.AUG_ROT_RANGE)
            aug_pts_rect = box_util.rotate_pc_along_y(aug_pts_rect, rot_angle=angle)
            if stage == 1:
                aug_gt_boxes3d = box_util.rotate_pc_along_y(aug_gt_boxes3d, rot_angle=angle)
            elif stage == 2:
                assert aug_gt_boxes3d.shape[0] == 2
                aug_gt_boxes3d[0] = self.rotate_box3d_along_y(aug_gt_boxes3d[0], angle)
                aug_gt_boxes3d[1] = self.rotate_box3d_along_y(aug_gt_boxes3d[1], angle)
            else:
                raise NotImplementedError

        if 'scaling' in aug_list and aug_enable[1] < cfg.AUG_METHOD_PROB[1]:
            scale = np.random.uniform(0.95, 1.05)
            aug_pts_rect = aug_pts_rect * scale
            aug_gt_boxes3d[:, 0:6] = aug_gt_boxes3d[:, 0:6] * scale

        if 'flip' in aug_list and aug_enable[2] < cfg.AUG_METHOD_PROB[2]:
            # flip horizontal
            aug_pts_rect[:, 0] = -aug_pts_rect[:, 0]
            aug_gt_boxes3d[:, 0] = -aug_gt_boxes3d[:, 0]
            # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
            if stage == 1:
                aug_gt_boxes3d[:, 6] = np.sign(aug_gt_boxes3d[:, 6]) * np.pi - aug_gt_boxes3d[:, 6]
            elif stage == 2:
                assert aug_gt_boxes3d.shape[0] == 2
                aug_gt_boxes3d[0, 6] = np.sign(aug_gt_boxes3d[0, 6]) * np.pi - aug_gt_boxes3d[0, 6]
                aug_gt_boxes3d[1, 6] = np.sign(aug_gt_boxes3d[1, 6]) * np.pi - aug_gt_boxes3d[1, 6]
            else:
                raise NotImplementedError

        aug_gt_boxes3d = np.squeeze(aug_gt_boxes3d)

        return aug_pts_rect, aug_gt_boxes3d

    @staticmethod
    def rotate_box3d_along_y(box3d, rot_angle) -> np.ndarray:
        old_x, old_z, ry = box3d[0], box3d[2], box3d[6]
        old_beta = np.arctan2(old_z, old_x)
        alpha = -np.sign(old_beta) * np.pi / 2 + old_beta + ry
        box3d = box_util.rotate_pc_along_y(box3d.reshape(1, 7), rot_angle=rot_angle)[0]
        new_x, new_z = box3d[0], box3d[2]
        new_beta = np.arctan2(new_z, new_x)
        box3d[6] = np.sign(new_beta) * np.pi / 2 + alpha - new_beta
        return box3d

    @staticmethod
    def generate_rpn_training_label(points, box) -> Tuple:
        """根据点云和真值框生成对应的伪label

        :param points: 4维的点云，　(x, y, z, seg)
        :param box: 真值框，　(x, y, z, w, h, l ,ry)  xyz是bbox中心坐标
        :return: cls_label, reg_label, seg_label
        """
        cls_label = np.zeros((points.shape[0]), dtype=np.int32)
        seg_label = np.zeros((points.shape[0]), dtype=np.int32)
        reg_label = np.zeros((points.shape[0], 7), dtype=np.float32)

        # covert box to corner
        box_corners = center_box_to_corners(box)    # (8, 3)
        # get flag of each point
        points_flag = box_util.in_hull(points[:, 0:3], box_corners)  # the order of pts  does not change
        fg_pts_rect = points[points_flag]   # fg_pts_rect.shape is (n, 4) [n < 500], points.shape is (500, 4)
        cls_label[points_flag] = 1  # shape is (500,)

        # enlarge box and label the ignore points
        extend_boxes3d = box_util.enlarge_box3d(box, extra_width=0.2)
        extend_corners = center_box_to_corners(extend_boxes3d)
        fg_enlarge_flag = box_util.in_hull(points[:, 0:3], extend_corners)
        ignore_flag = np.logical_xor(points_flag, fg_enlarge_flag)
        cls_label[ignore_flag] = -1

        reg_label[points_flag, 0:3] = box[:3] - fg_pts_rect[:, 0:3]
        reg_label[points_flag, 3] = box[3]
        reg_label[points_flag, 4] = box[4]
        reg_label[points_flag, 5] = box[5]
        reg_label[points_flag, 6] = box[6]

        seg_label = points[:, 3]

        return cls_label, reg_label, seg_label

    @staticmethod
    def template_point_prepare(template_points, template_box):
        template_extend_boxes3d = box_util.enlarge_box3d(template_box, extra_width=0.5)
        template_extend_extend_corners = center_box_to_corners(template_extend_boxes3d)
        points_flag = box_util.in_hull(template_points, template_extend_extend_corners)
        template_last_points = template_points[points_flag == True]

        return template_last_points


if __name__ == '__main__':
    dataset = PandaDataset('pandaset/dataset_test')
    dataset[19]




