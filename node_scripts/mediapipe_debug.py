#!/usr/bin/env python

import rospy
import cv2
import mediapipe as mp
import numpy as np
import tf2_ros
import geometry_msgs.msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R

class WristTFPublisher:
    def __init__(self):
        rospy.init_node('wrist_tf_publisher', anonymous=True)
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        # 订阅Realsense发布的彩色图像和深度图像
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)

        # 初始化深度图像
        self.depth_image = None

        # Realsense相机内参（需要根据实际相机校准参数填写）
        self.fx = 616.368  # 焦距 x
        self.fy = 616.745  # 焦距 y
        self.cx = 319.935  # 主点 x
        self.cy = 243.639  # 主点 y

    def depth_callback(self, msg):
        try:
            # 将深度图像消息转换为OpenCV格式
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def image_callback(self, msg):
        if self.depth_image is None:
            return

        try:
            # 将彩色图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # 转换图像为RGB
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 获取手腕位置（landmark 0）
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                h, w, c = cv_image.shape
                cx, cy = int(wrist.x * w), int(wrist.y * h)

                # 获取深度值
                depth = self.depth_image[cy, cx]  # 深度值（单位：毫米）

                # 将2D像素坐标 + 深度值转换为3D坐标
                if depth > 0:  # 确保深度值有效
                    x = (cx - self.cx) * depth / self.fx
                    y = (cy - self.cy) * depth / self.fy
                    z = depth

                    # 计算手腕的旋转
                    rotation = self.calculate_wrist_rotation(hand_landmarks)

                    # 发布TF
                    self.publish_tf(x, y, z, rotation)

                # 在图像上绘制手腕位置
                cv2.circle(cv_image, (cx, cy), 5, (255, 0, 0), -1)

        # 显示图像
        cv2.imshow('Wrist Detection', cv_image)
        cv2.waitKey(1)

    def calculate_wrist_rotation(self, hand_landmarks):
        # 获取手腕和中指根部的3D坐标
        wrist = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].z
        ])
        middle_mcp = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z
        ])

        # 计算手腕到中指根部的向量
        vector = middle_mcp - wrist
        vector = vector / np.linalg.norm(vector)  # 归一化

        # 计算旋转矩阵
        rotation = R.align_vectors([[1, 0, 0]], [vector])[0]  # 将向量对齐到X轴
        return rotation.as_quat()  # 返回四元数

    def publish_tf(self, x, y, z, rotation):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "camera_color_optical_frame"  # Realsense的彩色摄像头光学坐标系
        t.child_frame_id = "wrist_frame"
        t.transform.translation.x = x / 1000.0  # 转换为米
        t.transform.translation.y = y / 1000.0  # 转换为米
        t.transform.translation.z = z / 1000.0  # 转换为米
        t.transform.rotation.x = rotation[0]
        t.transform.rotation.y = rotation[1]
        t.transform.rotation.z = rotation[2]
        t.transform.rotation.w = rotation[3]
        self.tf_broadcaster.sendTransform(t)

if __name__ == '__main__':
    try:
        wrist_tf_publisher = WristTFPublisher()
        rospy.spin()  # 保持节点运行，等待回调
    except rospy.ROSInterruptException:
        pass
