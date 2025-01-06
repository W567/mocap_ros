#!/usr/bin/env python
import sys
import select
import numpy as np
from scipy.spatial.transform import Rotation as R

import tf
import rospy
from sensor_msgs.msg import JointState

from opt_ik import OptIK



def get_rot_angle(axis_orig, axis, plane_normal):
    axis_orig = axis_orig / np.linalg.norm(axis_orig)
    axis = axis / np.linalg.norm(axis)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    axis_proj = axis - np.dot(axis, plane_normal) * plane_normal
    return np.arccos(np.clip(np.dot(axis_orig, axis_proj), -1.0, 1.0))



class Tracker(OptIK):
    def __init__(self):
        super(Tracker, self).__init__()
        rospy.init_node('tracker', anonymous=True)

        rate = rospy.get_param('~rate', 30.0)
        self.tracker_rate = rospy.Rate(rate)

        self.sim = rospy.get_param('sim', True)
        self.tracking_hand = True
        self.tracking_finger = True

        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.right_hand_joint_publisher = rospy.Publisher('/joint_states', JointState, queue_size=1)

        self.prev_hand_qpos = np.zeros(22)
        self.ema_theta = 0.9
        if not self.sim:
            self.srh_joint_target = JointState()
            self.srh_joint_target.name = ['rh_WRJ2', 'rh_WRJ1',
                                'rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1',
                                'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1',
                                'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1',
                                'rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1',
                                'rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1',]
            self.srh_joint_target.position = np.zeros(24)
            self.srh_joint_target.velocity = np.zeros(24)
            self.srh_joint_target.effort = np.zeros(24)
            self.srh_ctrl_pub = rospy.Publisher("srh_joint_target", JointState, queue_size=1)

        rospy.loginfo("[Tracker] Initialized")


    def lookup_transform(self, target_frame, source_frame):
        try:
            (trans,rot) = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            # rospy.logerr("[Tracker] Failed to lookup transform")
            trans = None
            rot = None
        return trans, rot

    def track_body(self, publisher, body_name, base_name='/world'):
        trans, rot = self.lookup_transform(base_name, body_name)
        if trans is None or rot is None:
            return
        rotation = R.from_quat(rot)
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = trans

        if 'right_hand/wrist' in body_name:

            srh2mano = np.array([
                [-0.1254368, 0.07305742, -0.98940802, 0.00956725],
                [0.05917042, 0.99606057, 0.06604704, -0.01125753],
                [0.99033553, -0.05025896, -0.12926549, -0.00170861],
                [0.0, 0.0, 0.0, 1.0]
            ])

            srh_palm_tf = matrix @ srh2mano
            trans = srh_palm_tf[:3, 3]
            quat = R.from_matrix(srh_palm_tf[:3, :3]).as_quat()

            # publish tf of 'ik_palm' to be same with body_name relative to base_name
            self.tf_broadcaster.sendTransform(trans, quat, rospy.Time.now(), 'rh_palm', base_name)
        else:
            raise NotImplementedError("Only right hand is supported")


    def track_index(self):
        trans, rot = self.lookup_transform('rh_palm', 'right_hand/index3')
        if trans is None or rot is None:
            return
        self.ind_pos = np.array(trans)
        self.ind_nor = -R.from_quat(rot).as_matrix()[:, 1]
        self.ind_pos[1] *= 1.142

    def track_middle(self):
        trans, rot = self.lookup_transform('rh_palm', '/right_hand/middle3')
        if trans is None or rot is None:
            return
        self.mid_pos = np.array(trans)
        self.mid_nor = -R.from_quat(rot).as_matrix()[:, 1]
        self.mid_pos[1] *= 1.142

    def track_ring(self):
        trans, rot = self.lookup_transform('rh_palm', '/right_hand/ring3')
        if trans is None or rot is None:
            return
        self.rin_pos = np.array(trans)
        self.rin_nor = -R.from_quat(rot).as_matrix()[:, 1]
        self.rin_pos[1] *= 1.142

    def track_pinky(self):
        trans, rot = self.lookup_transform('rh_palm', '/right_hand/pinky3')
        if trans is None or rot is None:
            return
        self.lit_pos = np.array(trans)
        self.lit_nor = -R.from_quat(rot).as_matrix()[:, 1]
        self.lit_pos[1] *= 1.142

    def track_thumb(self):
        trans, rot = self.lookup_transform('rh_palm', '/right_hand/thumb3')
        if trans is None or rot is None:
            return
        self.thu_pos = np.array(trans)
        self.thu_nor = -R.from_quat(rot).as_matrix()[:, 1]
        self.thu_pos[1] *= 1.142


    def track_finger(self):
        self.track_index()
        self.track_middle()
        self.track_ring()
        self.track_pinky()
        self.track_thumb()

        # print("===========================================")
        # print("thumb")
        # print(np.array2string(self.thu_pos, separator=', '))
        # print(np.array2string(self.thu_nor, separator=', '))
        # print("index")
        # print(np.array2string(self.ind_pos, separator=', '))
        # print(np.array2string(self.ind_nor, separator=', '))
        # print("middle")
        # print(np.array2string(self.mid_pos, separator=', '))
        # print(np.array2string(self.mid_nor, separator=', '))
        # print("ring")
        # print(np.array2string(self.rin_pos, separator=', '))
        # print(np.array2string(self.rin_nor, separator=', '))
        # print("little")
        # print(np.array2string(self.lit_pos, separator=', '))
        # print(np.array2string(self.lit_nor, separator=', '))

        hand_qpos = self.optimize()

        ema_hand_qpos = self.ema_theta * self.prev_hand_qpos + (1 - self.ema_theta) * hand_qpos
        self.prev_hand_qpos = ema_hand_qpos

        if not self.sim:
            self.srh_joint_target.position = np.concatenate([np.zeros(2), ema_hand_qpos])
            self.srh_ctrl_pub.publish(self.srh_joint_target)
        else:
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = ['rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1',
                        'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1',
                        'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1',
                        'rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1',
                        'rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1']

            msg.position = ema_hand_qpos
            self.right_hand_joint_publisher.publish(msg)



    def execute(self):
        while not rospy.is_shutdown():
            if select.select([sys.stdin,],[],[],0.0) == ([sys.stdin],[],[]):
                user_input = sys.stdin.readline().strip()
                if user_input == 'h':
                    self.tracking_hand = not self.tracking_hand
                    if self.tracking_hand:
                        rospy.loginfo("[Tracker] Start tracking hand")
                    else:
                        rospy.loginfo("[Tracker] Stop tracking hand")
                elif user_input == 'f':
                    self.tracking_finger = not self.tracking_finger
                    if self.tracking_finger:
                        rospy.loginfo("[Tracker] Start tracking finger")
                    else:
                        rospy.loginfo("[Tracker] Stop tracking finger")
                elif user_input == 'q':
                    rospy.loginfo("[Tracker] Quit")
                    break

            if self.tracking_hand:
                self.track_body(None, 'right_hand/wrist', base_name='/camera_color_optical_frame')
            if self.tracking_finger:
                self.track_finger()

            self.tracker_rate.sleep()


def main():
    handle = Tracker()
    handle.execute()

if __name__ == '__main__':
    main()
