#!/usr/bin/env python
import tf
import rospy
import numpy as np
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


def create_arrow_array():
    rospy.init_node('tip_arrow_publisher', anonymous=True)
    marker_pub = rospy.Publisher('/tip_arrow', MarkerArray, queue_size=1)

    rate_num = rospy.get_param("tip_arrow_rate", 15)
    root_frame = rospy.get_param("root_frame", "camera_link")
    rate = rospy.Rate(rate_num)

    namespace = rospy.get_namespace()
    if namespace == '/':
        namespace = ''
    elif namespace[-1] != '/':
        namespace += '/'

    frame_names = [
        "right_hand/index3",
        "right_hand/middle3",
        "right_hand/pinky3",
        "right_hand/ring3",
        "right_hand/thumb3",
        "left_hand/index3",
        "left_hand/middle3",
        "left_hand/pinky3",
        "left_hand/ring3",
        "left_hand/thumb3"
    ]
    frame_names = [namespace + name for name in frame_names]

    start_point = Point(x=0.0, y=0.0, z=0.0)

    arrow_end_points = {
        "right_thumb": np.array([-0.47075654, -0.55634864, -0.68473679]) / 50.0,
        "right_finger": np.array([0, -1, 0]) / 50.0,
        "left_thumb": np.array([0.47075654, 0.55634864, 0.68473679]) / 50.0,
        "left_finger": np.array([0, 1, 0]) / 50.0,
    }

    tf_listener = tf.TransformListener()

    while not rospy.is_shutdown():
        marker_array = MarkerArray()

        for i, frame_name in enumerate(frame_names):
            marker = Marker()
            marker.header.frame_id = frame_name
            marker.header.stamp = rospy.Time.now()

            marker.ns = "tip_arrow_array"
            marker.id = i
            marker.type = Marker.ARROW

            marker.pose.orientation.w = 1.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0

            if tf_listener.canTransform(frame_name, root_frame, rospy.Time(0)):
                marker.action = Marker.ADD

                end_point = Point()
                if 'right' in frame_name:
                    if 'thumb' in frame_name:
                        arrow_end = arrow_end_points["right_thumb"]
                    else:
                        arrow_end = arrow_end_points["right_finger"]
                else:
                    if 'thumb' in frame_name:
                        arrow_end = arrow_end_points["left_thumb"]
                    else:
                        arrow_end = arrow_end_points["left_finger"]

                end_point.x, end_point.y, end_point.z = arrow_end
                marker.points = [start_point, end_point]

                marker.scale.x = 0.005
                marker.scale.y = 0.005
                marker.scale.z = 0
                marker.color = ColorRGBA(1.0, 1.0, 0.0, 1.0)
            else:
                marker.action = Marker.DELETE

            marker.lifetime = rospy.Duration()
            marker_array.markers.append(marker)

        marker_pub.publish(marker_array)
        rate.sleep()

if __name__ == '__main__':
    try:
        create_arrow_array()
    except rospy.ROSInterruptException:
        pass