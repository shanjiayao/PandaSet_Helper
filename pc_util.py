import os
import numpy as np
import pcl
import pcl.pcl_visualization


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


def pcl_visual_pc(points):
    # 归一化
    sum_x, sum_y, sum_z = 0, 0, 0
    # print(points[0])
    for i in range(points.shape[0]):
        sum_x += points[i][0]
        sum_y += points[i][0]
        sum_z += points[i][0]
    avg_x = sum_x / points.shape[0]
    avg_y = sum_y / points.shape[0]
    avg_z = sum_z / points.shape[0]
    for i in range(points.shape[0]):
        points[i][0] -= avg_x
        points[i][1] -= avg_y
        points[i][2] -= avg_z
    print(points[0])
    pc = np.zeros([points.shape[0], 4])
    #　上色
    pc[:, 3] = 255
    pc = np.float32(pc)
    color_cloud = pcl.PointCloud_PointXYZRGB(pc)
    visual = pcl.pcl_visualization.CloudViewing()
    visual.ShowColorCloud(color_cloud, b'cloud')
    flag = True
    while True:
        flag != visual.WasStopped()
        if flag is False:
            break


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