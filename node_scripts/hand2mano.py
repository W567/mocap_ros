import os
import yaml
import numpy as np
from urdfpy import URDF
from typing import List
from scipy.spatial.transform import Rotation as R

import skrobot
from skrobot.model.primitives import Axis
from skrobot.model.robot_model import RobotModel
from skrobot.coordinates.base import Coordinates
from skrobot.viewers import PyrenderViewer as Viewer
from skrobot.coordinates.math import quaternion2matrix

from bcolors import bcolors

import rospkg
ros_package = rospkg.RosPack()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--robot", type=str, default="srh_float")
parser.add_argument("--left", action="store_true", help="Use left hand")
args = parser.parse_args()


current_path = os.path.dirname(os.path.realpath(__file__))
cfg_folder_path = os.path.join(current_path, "cfg")
cfg_file_path = os.path.join(cfg_folder_path, f"{args.robot}.yaml")
if os.path.exists(cfg_file_path):
    with open(cfg_file_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        mano_tip_frames = cfg['mano_tip_frames']
        tip_frames = cfg['tip_frames']
        palm_frame = cfg['palm_frame']
        mano_base_links = cfg['mano_base_links']
        robot_base_links = cfg['robot_base_links']
        assert len(mano_base_links) == len(robot_base_links), \
            (f"{bcolors.OKYELLOW}The number of base links should be the same."
             f"Fix in {cfg_file_path}{bcolors.ENDC}")
else:
    print(f"{bcolors.OKYELLOW}Configuration file for {args.robot} does not exist.\n")
    print(f"Please run get_robot_cfg.py to generate the configuration file.{bcolors.ENDC}")
    exit()


def set_model_color(model, color):
    for link in model.link_list:
        try:
            link.set_color(color)
        except:
            pass


def set_robot_state(
        input_robot_model: RobotModel,
        joint_names: List[str],
        hand_pose: np.ndarray,
        palm_pose: np.ndarray,
) -> None:
    pos, quat = palm_pose[:3], palm_pose[3:]
    # From x, y, z, w to w, x, y, z
    quat = [quat[3], quat[0], quat[1], quat[2]]

    coordinate = Coordinates(pos=list(pos), rot=quaternion2matrix(quat))
    input_robot_model.newcoords(coordinate)

    for joint_name, angle in zip(joint_names, hand_pose):
        input_robot_model.__dict__[joint_name].joint_angle(angle)


def get_link_bases(input_robot_model, base_link_names):
    axes = []
    base_poses = []
    for i, link_name in enumerate(base_link_names):
        base_axis = Axis(axis_radius=0.001, axis_length=0.01)
        base_link = input_robot_model.__dict__[link_name]
        pos = base_link.worldcoords().translation
        quat = R.from_matrix(base_link.worldcoords().rotation).as_quat()
        quat = [quat[3], quat[0], quat[1], quat[2]]
        coordinate = Coordinates(pos=pos, rot=quaternion2matrix(quat))
        base_axis.newcoords(coordinate)
        axes.append(base_axis)
        base_poses.append(pos)
    return axes, np.array(base_poses)


def svd_based_transformation(source_points, target_points):
    """Compute the rigid transformation using SVD."""
    # Compute centroids
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    # Center the points
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # Compute the covariance matrix
    covariance_matrix = source_centered.T @ target_centered

    # Compute SVD
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Compute the optimal rotation matrix
    rotation_matrix = Vt.T @ U.T

    # Ensure a proper rotation (no reflection)
    if np.linalg.det(rotation_matrix) < 0:
        Vt[-1, :] *= -1
        rotation_matrix = Vt.T @ U.T

    # Compute the translation vector
    translation_vector = target_centroid - rotation_matrix @ source_centroid

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation_vector
    return transform_matrix


default_pose = np.array([0,0,0,0,0,0,1])
# We adopt the mano urdf model from: https://github.com/YuyangLee/mano-urdf.git
mano_model_path = ros_package.get_path("robot_models") + "/mano/urdf/mano.urdf"
mano_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file=str(mano_model_path))
set_model_color(mano_model, [0.5, 0.5, 0.0, 0.5])
mano_joint_names = [mano_model.joint_list[i].name for i in range(len(mano_model.joint_list))]
mano_joint_angles = np.zeros(len(mano_joint_names))

# 0, 1, 2, 3,     # index
# 4, 5, 6, 7,     # middle
# 8, 9, 10, 11,   # ring
# 12, 13, 14, 15, # little
# 16, 17, 18, 19  # thumb
set_robot_state(mano_model, mano_joint_names, mano_joint_angles, default_pose)

mano_axes, mano_base_poses = get_link_bases(mano_model, mano_base_links)

mano_link_names = [mano_model.link_list[i].name for i in range(len(mano_model.link_list))]
mano_tip_ids = [mano_link_names.index(link) for link in mano_tip_frames]
mano_tip_xyzs = np.array([mano_model.link_list[i].worldcoords().translation for i in mano_tip_ids])
mano_tip_rots = np.array([mano_model.link_list[i].worldcoords().rotation for i in mano_tip_ids])
mano_tip_axes = []

shifts = -np.array([[-0.026, -0.003, 0.024],
                    [-0.026, 0.001, -0.000],
                    [-0.026, 0.004, -0.004],
                    [-0.024, 0.004, -0.007],
                    [-0.018, 0.002, -0.010]])
# The current mano urdf has no tip frames, has to shift from the distal joint (link)
original_frames_for_shift = ["thumb3", "index3", "middle3", "ring3", "pinky3"]
reorder_indices = [original_frames_for_shift.index(frame) for frame in mano_tip_frames]
shifts = shifts[reorder_indices]

for i, (xyz, rot) in enumerate(zip(mano_tip_xyzs, mano_tip_rots)):
    tip_axis = Axis(axis_radius=0.001, axis_length=0.01)

    shift = shifts[i]
    dist = np.linalg.norm(np.array(shift))
    xyz = xyz + rot @ [0, 0, -dist]

    co = Coordinates(pos=xyz, rot=rot)
    tip_axis.newcoords(co)
    mano_tip_axes.append(tip_axis)



robot_model_path = ros_package.get_path("robot_models") + f"/{args.robot}/urdf/{args.robot}.urdf"

urdf_model = URDF.load(robot_model_path)
urdf_joint_names = [urdf_model.joints[i].name for i in range(len(urdf_model.joints))]

## If mimic joint exists in the robot urdf model, when setting joint angles using skrobot,
## mimic multiplier and offset should be considered.
urdf_mimic_multiplier = []
urdf_mimic_offset = []
for joint in urdf_model.joints:
    if joint.mimic is not None:
        urdf_mimic_multiplier.append(joint.mimic.multiplier)
        urdf_mimic_offset.append(joint.mimic.offset)
    else:
        urdf_mimic_multiplier.append(1.0)
        urdf_mimic_offset.append(0.0)


robot_model = skrobot.models.urdf.RobotModelFromURDF(urdf_file=str(robot_model_path))
set_model_color(robot_model, [0.5, 0.5, 0.5, 0.5])
robot_joint_names = [robot_model.joint_list[i].name for i in range(len(robot_model.joint_list))]
robot_joint_angles = np.zeros(len(robot_joint_names))

mimic_multiplier = []
mimic_offset = []
for joint_name in robot_joint_names:
    index = urdf_joint_names.index(joint_name)
    mimic_multiplier.append(urdf_mimic_multiplier[index])
    mimic_offset.append(urdf_mimic_offset[index])
mimic_multiplier = np.array(mimic_multiplier)
mimic_offset = np.array(mimic_offset)

robot_joint_angles = mimic_multiplier * robot_joint_angles + mimic_offset

set_robot_state(robot_model, robot_joint_names, robot_joint_angles, default_pose)
_, robot_base_poses = get_link_bases(robot_model, robot_base_links)

robot_tip_links = tip_frames
link_names = [robot_model.link_list[i].name for i in range(len(robot_model.link_list))]
tip_ids = [link_names.index(link) for link in robot_tip_links]


if args.left:
    # The mano urdf model is right hand, so we need to flip mano_base_poses on all axes
    mano_base_poses[:, 0] *= -1
    mano_base_poses[:, 1] *= -1
    mano_base_poses[:, 2] *= -1


# move robot_base_poses to mano_base_poses
hand2mano = svd_based_transformation(robot_base_poses, mano_base_poses)
print(f"Transform matrix for {args.robot} to have its root joints of fingers mapped to that of the mano hand:")
print(f"{hand2mano}\n")

trans = hand2mano[:3, 3]
rot = hand2mano[:3, :3]
mapped_palm_pos = np.array(trans)
mapped_palm_quat = R.from_matrix(rot).as_quat()
mapped_palm_pose = np.concatenate([mapped_palm_pos, mapped_palm_quat])

set_robot_state(robot_model, robot_joint_names, robot_joint_angles, mapped_palm_pose)
robot_axes, robot_base_poses = get_link_bases(robot_model, robot_base_links)

tip_xyzs = np.array([robot_model.link_list[i].worldcoords().translation for i in tip_ids])
tip_rots = np.array([robot_model.link_list[i].worldcoords().rotation for i in tip_ids])
tip_axes = []
for i, (xyz, rot) in enumerate(zip(tip_xyzs, tip_rots)):
    tip_axis = Axis(axis_radius=0.001, axis_length=0.01)
    co = Coordinates(pos=xyz, rot=rot)
    tip_axis.newcoords(co)
    tip_axes.append(tip_axis)


# Visualize the transformed robot model with mano hand model
viewer = Viewer()
viewer.add(robot_model)
viewer.add(mano_model)
axis = Axis(axis_radius=0.001, axis_length=0.05)
viewer.add(axis)
for axis in mano_axes:
    viewer.add(axis)
for axis in mano_tip_axes:
    viewer.add(axis)
for axis in robot_axes:
    viewer.add(axis)
for axis in tip_axes:
    viewer.add(axis)
mapped_robot_palm_axis = Axis(axis_radius=0.001, axis_length=0.025)
co = Coordinates(pos=mapped_palm_pos, rot=quaternion2matrix(mapped_palm_quat[[3, 0, 1, 2]]))
mapped_robot_palm_axis.newcoords(co)
viewer.add(mapped_robot_palm_axis)
viewer.show()


robot_length = []
for i, (robot_base_axis, robot_tip_axis) in enumerate(zip(robot_axes, tip_axes)):
    length = np.linalg.norm(robot_base_axis.worldcoords().translation - robot_tip_axis.worldcoords().translation)
    robot_length.append(length)
robot_length = np.array(robot_length)

mano_length = []
for i, (mano_base_axis, mano_tip_axis) in enumerate(zip(mano_axes, mano_tip_axes)):
    length = np.linalg.norm(mano_base_axis.worldcoords().translation - mano_tip_axis.worldcoords().translation)
    mano_length.append(length)
mano_length = np.array(mano_length)

print(f"{args.robot} finger lengths")
print(tip_frames)
print(np.array2string(robot_length, separator=', '))
print("mano hand finger lengths")
print(mano_tip_frames)
print(np.array2string(mano_length, separator=', '))

print(f" ============== finger length scale {args.robot} / mano ============== ")
print(robot_length / mano_length)

input(f"{bcolors.OKCYAN}Press any key to exit...{bcolors.ENDC}")

cfg['hand2mano'] = hand2mano.tolist()
with open(cfg_file_path, "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=None, sort_keys=False)


print(f"{bcolors.OKGREEN}Configuration file for {args.robot} is updated"
      f"with the hand2mano transform matrix.{bcolors.ENDC}")