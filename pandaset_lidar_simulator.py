from __future__ import print_function
import numpy as np

# ros
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.append(ros_path)
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import *
from std_msgs.msg import Header
from std_msgs.msg import Int64
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

# pandaset
from pandaset.dataset import DataSet
from pandaset.geometry import center_box_to_corners


"""
- Set for cuboids with `label` values in
    - _Car_
    - _Pickup Truck_
    - _Medium-sized Truck_
    - _Semi-truck_
    - _Towed Object_
    - _Motorcycle_
    - _Other Vehicle - Construction Vehicle_
    - _Other Vehicle - Uncommon_
    - _Other Vehicle - Pedicab_
    - _Emergency Vehicle_
    - _Bus_
    - _Personal Mobility Device_
    - _Motorized Scooter_
    - _Bicycle_
    - _Train_
    - _Trolley_
    - _Tram / Subway_
"""


class LidarSimulator:
    def __init__(self, path, seq_id) -> None:
        self.frame = 0
        self.dataset_path = path
        self.seq_num = seq_id
        self.pointcloud = None
        self.dataset = None
        self.cuboids_label = None
        self.seq = None
        self.lidar = None

        # ROS
        rospy.init_node('lidar_simulator', anonymous=True)
        self.rate = rospy.Rate(10)
        self.point_pub = rospy.Publisher('/kitti_points', PointCloud2, queue_size=1)
        self.frame_pub = rospy.Publisher('/frame_number', Int64, queue_size=1)
        self.label_pub = rospy.Publisher('/label_bbox', MarkerArray, queue_size=1)

        # Init
        self.get_dataset(self.dataset_path)
        self.load_sequences(with_semseg=True, seq_num=self.seq_num)
        self.load_lidar()
        self.load_cuboids_label()

    def get_dataset(self, path):
        self.dataset = DataSet(path)

    def load_sequences(self, seq_num, with_semseg=True):
        seg_seq = self.dataset.sequences(with_semseg=with_semseg)
        self.seq = self.dataset[seg_seq[seq_num]]
        self.seq.load()

    def load_lidar(self):
        self.lidar = self.seq.lidar.data

    def load_cuboids_label(self):
        self.cuboids_label = self.seq.cuboids.data

    def get_frame_points(self, frame):
        self.frame = frame
        self.pointcloud = self.lidar[frame].to_numpy()[:, :3]

    def show_bbox_label(self, classes='Car'):
        # 读取label，序号为当前帧号
        box_label = self.cuboids_label[self.frame].to_numpy()
        all_bbox = MarkerArray()
        for line_number in range(box_label.shape[0]):
            self.bbox_data = np.zeros([box_label.shape[0], 9, 3])

            if box_label[line_number][1] == classes:
                # get bbox data
                box = []
                box = box_label[line_number][5:11]
                box = box.tolist()
                box.append(box_label[line_number][2])
                corners = center_box_to_corners(box)

                # Point to Marker
                point_0 = Point(corners[0][0], corners[0][1], corners[0][2])
                point_1 = Point(corners[1][0], corners[1][1], corners[1][2])
                point_2 = Point(corners[2][0], corners[2][1], corners[2][2])
                point_3 = Point(corners[3][0], corners[3][1], corners[3][2])
                point_4 = Point(corners[4][0], corners[4][1], corners[4][2])
                point_5 = Point(corners[5][0], corners[5][1], corners[5][2])
                point_6 = Point(corners[6][0], corners[6][1], corners[6][2])
                point_7 = Point(corners[7][0], corners[7][1], corners[7][2])

                # marker
                marker = Marker(id=line_number)
                marker.type = Marker.LINE_LIST
                marker.ns = 'velodyne'
                marker.action = Marker.ADD
                marker.header.frame_id = "/velodyne"
                marker.header.stamp = rospy.Time.now()

                marker.points.append(point_1)
                marker.points.append(point_2)
                marker.points.append(point_1)
                marker.points.append(point_0)
                marker.points.append(point_1)
                marker.points.append(point_5)
                marker.points.append(point_7)
                marker.points.append(point_4)
                marker.points.append(point_7)
                marker.points.append(point_6)
                marker.points.append(point_7)
                marker.points.append(point_3)
                marker.points.append(point_2)
                marker.points.append(point_6)
                marker.points.append(point_2)
                marker.points.append(point_3)
                marker.points.append(point_0)
                marker.points.append(point_4)
                marker.points.append(point_0)
                marker.points.append(point_3)
                marker.points.append(point_5)
                marker.points.append(point_6)
                marker.points.append(point_5)
                marker.points.append(point_4)
                # ********************
                marker.lifetime = rospy.Duration.from_sec(0.2)
                marker.text = str(1)
                marker.scale.x = 0.05
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                all_bbox.markers.append(marker)
        print("publish label bbox over")
        self.label_pub.publish(all_bbox)

    def publish_pointcloud(self):
        # pointcloud pub
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'velodyne'
        cloud_msg = pc2.create_cloud_xyz32(header, self.pointcloud)
        self.point_pub.publish(cloud_msg)
        print('publish_pointcloud')

    def spin(self, start_frame, classes='Car', is_run=False):
        while not rospy.is_shutdown():
            if is_run is False:
                self.get_frame_points(start_frame)
                self.publish_pointcloud()
                self.show_bbox_label(classes=classes)
                self.rate.sleep()
            else:
                for i in range(len(self.lidar)):
                    self.get_frame_points(i)
                    self.publish_pointcloud()
                    self.show_bbox_label(classes=classes)
                    self.rate.sleep()


if __name__ == '__main__':
    path = '/media/echo/仓库卷/DataSet/Autonomous Driving/PandaSets/pandaset'
    seq_id = 0
    lidar = LidarSimulator(path, seq_id)
    # is_run = False 只读取对应start_frame帧的点云和label，　若为true,则循环这一序列的所有帧
    lidar.spin(start_frame=0, classes='Bicycle', is_run=True)

"""
seq = 0
    'Car' 'Pickup Truck' 'Motorcycle'

"""