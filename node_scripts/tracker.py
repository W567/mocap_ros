#!/usr/bin/env python
import os
import sys
import yaml
import select
import numpy as np
from scipy.spatial.transform import Rotation as R

import tf
import rospy
from sensor_msgs.msg import JointState

from opt_ik import OptIK



class Tracker(OptIK):
    def __init__(self):
        rospy.init_node('tracker', anonymous=True)

        robot = rospy.get_param('robot', 'srh_float')
        tol = rospy.get_param('~tol', 1e-4)
        collision_threshold = rospy.get_param('~collision_threshold', 0.018)
        nor_weight = rospy.get_param('~nor_weight', 0.0)
        col_weight = rospy.get_param('~col_weight', 1.0)
        verbose = rospy.get_param('~verbose', False)
        with_collision = rospy.get_param('~with_collision', True)

        OptIK.__init__(
            self,
            robot=robot,
            tol=tol,
            nor_weight=nor_weight,
            col_weight=col_weight,
            collision_threshold=collision_threshold,
            verbose=verbose,
            with_collision=with_collision
        )

        rate = rospy.get_param('~rate', 30.0)
        self.tracker_rate = rospy.Rate(rate)

        self.sim = rospy.get_param('sim', True)
        self.tracking_hand = True
        self.tracking_finger = True

        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.right_hand_joint_publisher = rospy.Publisher('/joint_states', JointState, queue_size=1)

        # Mano hand tip normal (pulp direction) under tip local frame
        self.mano_thu_nor = np.array([-0.47075654, -0.55634864, -0.68473679])
        self.mano_fin_nor = np.array([0., -1., 0.])
        self.mano_thu_nor /= np.linalg.norm(self.mano_thu_nor)
        self.mano_fin_nor /= np.linalg.norm(self.mano_fin_nor)

        current_path = os.path.dirname(os.path.realpath(__file__))
        cfg_file_path = os.path.join(current_path, "cfg", f"{robot}.yaml")
        with open(cfg_file_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
            self.urdf_joint_names = cfg['urdf_joint_names']
            self.palm_frame = cfg['palm_frame']
            self.hand2mano = cfg['hand2mano']
            self.hand2mano = np.array(self.hand2mano).reshape(4, 4)

        self.prev_hand_qpos = np.zeros(len(self.urdf_joint_names))
        self.finger_ema_alpha = rospy.get_param('~finger_ema_alpha', 0.9)
        if not self.sim:
            # TODO not for sure if WRJ are required or not
            self.srh_joint_target = JointState()
            self.srh_joint_target.name = self.urdf_joint_names
            self.srh_joint_target.position = np.zeros(len(self.urdf_joint_names))
            self.srh_joint_target.velocity = np.zeros(len(self.urdf_joint_names))
            self.srh_joint_target.effort = np.zeros(len(self.urdf_joint_names))
            self.srh_ctrl_pub = rospy.Publisher("srh_joint_target", JointState, queue_size=1)


        self.track_finger_methods = [
            self.track_thumb,
            self.track_index,
            self.track_middle,
            self.track_ring,
            self.track_pinky,
        ]
        # Here, we assume the fingers are disabled in a fixed order.
        self.track_finger_methods = self.track_finger_methods[:self.num_fingers]

        rospy.loginfo("[Tracker] Initialized")


    def lookup_transform(self, target_frame, source_frame):
        try:
            (trans,rot) = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            trans = None
            rot = None
        return trans, rot

    def track_body(self, base_name='/world'):
        trans, rot = self.lookup_transform(base_name, 'right_hand/wrist')
        if trans is None or rot is None:
            return
        rotation = R.from_quat(rot)
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = trans

        hand_palm_tf = matrix @ self.hand2mano
        trans = hand_palm_tf[:3, 3]
        quat = R.from_matrix(hand_palm_tf[:3, :3]).as_quat()
        self.tf_broadcaster.sendTransform(trans, quat, rospy.Time.now(), self.palm_frame, base_name)

    def track_index(self):
        trans, rot = self.lookup_transform(self.palm_frame, 'right_hand/index3')
        if trans is None or rot is None:
            return
        self.index_pos = np.array(trans)
        self.index_nor = R.from_quat(rot).as_matrix() @ self.mano_fin_nor
        self.index_pos[1] *= 1.142

    def track_middle(self):
        trans, rot = self.lookup_transform(self.palm_frame, '/right_hand/middle3')
        if trans is None or rot is None:
            return
        self.middle_pos = np.array(trans)
        self.middle_nor = R.from_quat(rot).as_matrix() @ self.mano_fin_nor
        self.middle_pos[1] *= 1.142

    def track_ring(self):
        trans, rot = self.lookup_transform(self.palm_frame, '/right_hand/ring3')
        if trans is None or rot is None:
            return
        self.ring_pos = np.array(trans)
        self.ring_nor = R.from_quat(rot).as_matrix() @ self.mano_fin_nor
        self.ring_pos[1] *= 1.142

    def track_pinky(self):
        trans, rot = self.lookup_transform(self.palm_frame, '/right_hand/pinky3')
        if trans is None or rot is None:
            return
        self.pinky_pos = np.array(trans)
        self.pinky_nor = R.from_quat(rot).as_matrix() @ self.mano_fin_nor
        self.pinky_pos[1] *= 1.142

    def track_thumb(self):
        trans, rot = self.lookup_transform(self.palm_frame, '/right_hand/thumb3')
        if trans is None or rot is None:
            return
        self.thumb_pos = np.array(trans)
        self.thumb_nor = R.from_quat(rot).as_matrix() @ self.mano_thu_nor
        self.thumb_pos[1] *= 1.142


    def track_finger(self):
        for method in self.track_finger_methods:
            method()

        # print("===========================================")
        # print("thumb")
        # print(np.array2string(self.thumb_pos, separator=', '))
        # print(np.array2string(self.thumb_nor, separator=', '))
        # print("index")
        # print(np.array2string(self.index_pos, separator=', '))
        # print(np.array2string(self.index_nor, separator=', '))
        # print("middle")
        # print(np.array2string(self.middle_pos, separator=', '))
        # print(np.array2string(self.middle_nor, separator=', '))
        # print("ring")
        # print(np.array2string(self.ring_pos, separator=', '))
        # print(np.array2string(self.ring_nor, separator=', '))
        # print("little")
        # print(np.array2string(self.pinky_pos, separator=', '))
        # print(np.array2string(self.pinky_nor, separator=', '))

        hand_qpos = self.optimize()

        ema_hand_qpos = self.finger_ema_alpha * self.prev_hand_qpos + (1 - self.finger_ema_alpha) * hand_qpos
        self.prev_hand_qpos = ema_hand_qpos

        if not self.sim:
            self.srh_joint_target.position = np.concatenate([np.zeros(2), ema_hand_qpos])
            self.srh_ctrl_pub.publish(self.srh_joint_target)
        else:
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = self.urdf_joint_names

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
                self.track_body(base_name='/camera_color_optical_frame')
            if self.tracking_finger:
                self.track_finger()

            self.tracker_rate.sleep()


def main():
    handle = Tracker()
    handle.execute()

if __name__ == '__main__':
    main()
