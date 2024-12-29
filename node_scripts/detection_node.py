#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge, CvBridgeError
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Pose, Quaternion
from jsk_recognition_msgs.msg import Rect, HumanSkeleton, Segment
from mocap_ros.msg import Detection, DetectionArray, Mocap, MocapArray

from motion_capture.detector import DetectionModelFactory
from motion_capture.mocap import MocapModelFactory
from motion_capture.utils.utils import (
    MANO_KEYPOINT_NAMES,
    MANO_JOINTS_CONNECTION,
    MANO_CONNECTION_NAMES,
    SPIN_KEYPOINT_NAMES,
    SPIN_JOINTS_CONNECTION,
    SPIN_CONNECTION_NAMES,
)


def angle_between_planes(normal1, normal2):
    dot_product = np.dot(normal1, normal2)
    norm1 = np.linalg.norm(normal1)
    norm2 = np.linalg.norm(normal2)
    cos_theta = dot_product / (norm1 * norm2)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return theta


class DetectionNode(object):
    def __init__(self):
        self.mocap_model = None
        self.detection_model = None
        self.device = rospy.get_param("~device", "cuda:0")
        self.publish_tf = rospy.get_param("~publish_tf", True)
        self.camera_info = rospy.wait_for_message("~camera_info", CameraInfo)
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(self.camera_info)
        self.img_size = (self.camera_info.width, self.camera_info.height)

        self.camera_scale = rospy.get_param("~scale", 0.001)

        self.wrist_a = rospy.get_param("~wrist_a", 0.03)
        self.wrist_b = rospy.get_param("~wrist_b", 0.02)

        self.ema_alpha = rospy.get_param("~ema_alpha", 0.1)
        self.prev_wrist = None

        if self.publish_tf:
            self.tf_broadcaster = tf.TransformBroadcaster()

        # Detector
        self.detector = rospy.get_param(
            "~detector_model", "hand_object_detector"
        )  # hand_object_detector, mediapipe_hand
        if self.detector == "hand_object_detector":
            self.detector_config = {
                "threshold": rospy.get_param("~threshold", 0.9),
                "object_threshold": rospy.get_param("~object_threshold", 0.9),
                "margin": rospy.get_param("~margin", 10),
                "device": self.device,
            }
        elif self.detector == "mediapipe_hand":
            self.detector_config = {
                "threshold": rospy.get_param("~threshold", 0.9),
                "margin": rospy.get_param("~margin", 10),
                "device": self.device,
            }
        elif self.detector == "yolo":
            self.detector_config = {
                "margin": rospy.get_param("~margin", 10),
                "threshold": rospy.get_param("~threshold", 0.9),
                "device": self.device,
            }
        else:
            raise ValueError(f"Invalid detector model: {self.detector}")

        # Mocap
        self.with_mocap = rospy.get_param("~with_mocap", True)
        if self.with_mocap:
            self.mocap = rospy.get_param("~mocap_model", "hamer")  # frankmocap_hand, hamer, 4d-human
            self.with_visualize = rospy.get_param("~visualize", True)
            if self.mocap == "frankmocap_hand":
                self.mocap_config = {
                    "render_type": rospy.get_param("~render_type", "opengl"),  # pytorch3d, opendr, opengl
                    "img_size": self.img_size,
                    "visualize": self.with_visualize,
                    "device": self.device,
                }
                self.keypoint_names = MANO_KEYPOINT_NAMES
                self.connection_names = MANO_CONNECTION_NAMES
                self.joint_connections = MANO_JOINTS_CONNECTION
            elif self.mocap == "hamer":
                self.mocap_config = {
                    "focal_length": self.camera_model.fx(),
                    "rescale_factor": rospy.get_param("~rescale_factor", 2.0),
                    "img_size": self.img_size,
                    "visualize": self.with_visualize,
                    "device": self.device,
                }
                self.keypoint_names = MANO_KEYPOINT_NAMES
                self.connection_names = MANO_CONNECTION_NAMES
                self.joint_connections = MANO_JOINTS_CONNECTION
            elif self.mocap == "4d-human":
                self.mocap_config = {
                    "focal_length": self.camera_model.fx(),
                    "rescale_factor": rospy.get_param("~rescale_factor", 2.0),
                    "img_size": self.img_size,
                    "visualize": self.with_visualize,
                    "device": self.device,
                }
                self.keypoint_names = SPIN_KEYPOINT_NAMES
                self.connection_names = SPIN_CONNECTION_NAMES
                self.joint_connections = SPIN_JOINTS_CONNECTION
            else:
                raise ValueError(f"Invalid mocap model: {self.mocap}")

        # Initialize models and ROS
        self.bridge = CvBridge()
        self.init_model()
        self.sub = rospy.Subscriber("~input_image", Image, self.callback_image, queue_size=1, buff_size=2**24)

        self.depth_sub = rospy.Subscriber("~input_depth", Image, self.callback_depth, queue_size=1, buff_size=2 ** 24)
        self.depth_image = None

        self.pub_debug_image = rospy.Publisher("~debug_image", Image, queue_size=1)
        self.pub_detections = rospy.Publisher("~detections", DetectionArray, queue_size=1)
        if self.with_mocap:
            self.pub_mocaps = rospy.Publisher("~mocaps", MocapArray, queue_size=1)

    def init_model(self):
        self.detection_model = DetectionModelFactory.from_config(
            model=self.detector,
            model_config=self.detector_config,
        )

        if self.with_mocap:
            self.mocap_model = MocapModelFactory.from_config(
                model=self.mocap,
                model_config=self.mocap_config,
            )

    def callback_depth(self, msg):
        try:
            # Convert the depth image to a Numpy array
            cv_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except CvBridgeError as e:
            rospy.logerr(e)
            cv_image = None

        self.depth_image = cv_image

    def callback_image(self, msg):
        if self.depth_image is None:
            rospy.logwarn("Depth image is not received yet.")
            return
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        detections, visualization = self.detection_model.predict(image)

        # to DetectionArray msg
        detection_array = DetectionArray(header=msg.header)
        detection_array.detections = [
            Detection(
                label=detection.label,
                score=detection.score,
                rect=Rect(
                    x=int(detection.rect[0]),
                    y=int(detection.rect[1]),
                    width=int(detection.rect[2] - detection.rect[0]),
                    height=int(detection.rect[3] - detection.rect[1]),
                ),
            )
            for detection in detections
        ]
        self.pub_detections.publish(detection_array)

        if self.with_mocap:
            mocaps, visualization = self.mocap_model.predict(detections, image, visualization)
            # to MocapArray msg
            mocap_array = MocapArray(header=msg.header)
            mocap_array.mocaps = [
                Mocap(detection=detection_array.detections[i]) for i in range(len(detection_array.detections))
            ]

            for i in range(len(mocaps)):
                skeleton = HumanSkeleton()
                skeleton.bone_names = []
                skeleton.bones = []
                for j, (start, end) in enumerate(self.joint_connections):
                    bone = Segment()
                    bone.start_point = Point(
                        x=mocaps[i].keypoints[start][0],
                        y=mocaps[i].keypoints[start][1],
                        z=mocaps[i].keypoints[start][2],
                    )
                    bone.end_point = Point(
                        x=mocaps[i].keypoints[end][0],
                        y=mocaps[i].keypoints[end][1],
                        z=mocaps[i].keypoints[end][2],
                    )
                    skeleton.bones.append(bone)
                    skeleton.bone_names.append(self.connection_names[j])

                mocap_array.mocaps[i].pose = Pose(
                    position=Point(
                        x=mocaps[i].position[0],
                        y=mocaps[i].position[1],
                        z=mocaps[i].position[2],
                    ),
                    orientation=Quaternion(
                        w=mocaps[i].orientation[0],
                        x=mocaps[i].orientation[1],
                        y=mocaps[i].orientation[2],
                        z=mocaps[i].orientation[3],
                    ),
                )
                mocap_array.mocaps[i].skeleton = skeleton
            self.pub_mocaps.publish(mocap_array)
        else:
            mocap_array = MocapArray()

        # Publish visualization
        if self.with_visualize:
            vis_msg = self.bridge.cv2_to_imgmsg(visualization.astype(np.uint8), encoding="bgr8")
            vis_msg.header = msg.header
            self.pub_debug_image.publish(vis_msg)

        # Publish tf
        if self.publish_tf and self.with_mocap:
            for mocap in mocap_array.mocaps:
                try:
                    # predicted wrist pose in the camera frame
                    point_3d = (
                        mocap.pose.position.x,
                        mocap.pose.position.y,
                        mocap.pose.position.z,
                    )
                    # pixel coordinates of the wrist
                    point_2d = self.camera_model.project3dToPixel(point_3d)
                    # clip
                    point_2d = (
                        min(max(point_2d[0], 0), self.depth_image.shape[1] - 1),
                        min(max(point_2d[1], 0), self.depth_image.shape[0] - 1),
                    )
                    depth = self.depth_image[int(point_2d[1]), int(point_2d[0])]
                    if np.isnan(depth) or (depth == 0.0):
                        if self.prev_wrist is None:
                            continue
                        else:
                            wrist_cam = self.prev_wrist
                    else:
                        # Calculate 3D coordinates in the camera frame
                        x_cam = (point_2d[0] - self.camera_model.cx()) * depth / self.camera_model.fx() * self.camera_scale
                        y_cam = (point_2d[1] - self.camera_model.cy()) * depth / self.camera_model.fy() * self.camera_scale
                        z_cam = depth * self.camera_scale

                        wrist_quat = np.array([mocap.pose.orientation.x,
                                               mocap.pose.orientation.y,
                                               mocap.pose.orientation.z,
                                               mocap.pose.orientation.w])
                        wrist_rot = R.from_quat(wrist_quat).as_matrix()

                        # wrist eclipse intersection plane
                        wrist_plane_normal = wrist_rot[:, 0]
                        wrist_y_axis = wrist_rot[:, 1]
                        wrist_z_axis = wrist_rot[:, 2]

                        wrist_cam_orig = np.array([x_cam, y_cam, z_cam])
                        # view vector from the camera to the wrist
                        view_vector = wrist_cam_orig / np.linalg.norm(wrist_cam_orig)
                        # project the view vector to the wrist ellipse plane
                        view_vector_proj = view_vector - np.dot(view_vector, wrist_plane_normal) * wrist_plane_normal
                        view_vector_proj /= np.linalg.norm(view_vector_proj)

                        intersection_angle = np.arccos(np.abs(np.dot(wrist_z_axis, view_vector_proj)))
                        # slope of incident ray (perpendicular to the tangent of the ellipse at the intersection point (x0, y0))
                        view_k = np.tan(intersection_angle)

                        c = view_k * self.wrist_b ** 2 / self.wrist_a ** 2
                        # attention that x0 is along the z-axis of the wrist ellipse
                        x0 = np.sqrt(self.wrist_a ** 2 * self.wrist_b ** 2 / (self.wrist_b ** 2 + self.wrist_a ** 2 * c ** 2))
                        # y0 is along the y-axis of the wrist ellipse
                        y0 = c * x0

                        # determine the sign of x0 and y0 (quadrant of (x0, y0) in the wrist ellipse frame)
                        if np.dot(view_vector_proj, wrist_z_axis) < 0:
                            x0 = -x0
                        if np.dot(view_vector_proj, wrist_y_axis) < 0:
                            y0 = -y0

                        # translation to the wrist ellipse center
                        wrist_cam_new = wrist_cam_orig + wrist_rot @ np.array([0, y0, x0])

                        if self.prev_wrist is None:
                            wrist_cam = wrist_cam_new
                        else:
                            # Exponential Moving Average (EMA) filter to avoid serious jitter and deal with temporary occlusion
                            wrist_cam = self.ema_alpha * wrist_cam_new + (1 - self.ema_alpha) * self.prev_wrist
                        self.prev_wrist = wrist_cam

                    self.tf_broadcaster.sendTransform(
                        (
                            wrist_cam[0],
                            wrist_cam[1],
                            wrist_cam[2],
                        ),
                        (
                            mocap.pose.orientation.x,
                            mocap.pose.orientation.y,
                            mocap.pose.orientation.z,
                            mocap.pose.orientation.w,
                        ),
                        rospy.Time.now(),
                        mocap.detection.label + "/" + self.keypoint_names[0],
                        msg.header.frame_id,
                    )
                    # publish finger bone pose in the camera frame
                    for bone_name in mocap.skeleton.bone_names:
                        parent_name = bone_name.split("->")[0]
                        child_name = bone_name.split("->")[1]
                        bone_idx = mocap.skeleton.bone_names.index(bone_name)

                        parent_point = mocap.skeleton.bones[bone_idx].start_point
                        child_point = mocap.skeleton.bones[bone_idx].end_point
                        parent_to_child = R.from_quat(
                            [
                                mocap.pose.orientation.x,
                                mocap.pose.orientation.y,
                                mocap.pose.orientation.z,
                                mocap.pose.orientation.w,
                            ]
                        ).inv().as_matrix() @ np.array(
                            [
                                child_point.x - parent_point.x,
                                child_point.y - parent_point.y,
                                child_point.z - parent_point.z,
                            ]
                        )  # cause the bone is in the camera frame

                        # broadcast bone pose in tree structure
                        self.tf_broadcaster.sendTransform(
                            (parent_to_child[0], parent_to_child[1], parent_to_child[2]),
                            (0, 0, 0, 1),
                            rospy.Time.now(),
                            mocap.detection.label + "/" + child_name,
                            mocap.detection.label + "/" + parent_name,
                        )
                except (
                    tf.LookupException,
                    tf.ConnectivityException,
                    tf.ExtrapolationException,
                ) as e:
                    rospy.logerr(e)


if __name__ == "__main__":
    rospy.init_node("detection_node")
    node = DetectionNode()
    rospy.loginfo("Detection node started")
    rospy.spin()
