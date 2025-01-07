import os
import yaml
import rospkg
import pinocchio as pin
from urdfpy import URDF

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--robot", type=str, default="srh_float")
args = parser.parse_args()


ros_package = rospkg.RosPack()
model_path = ros_package.get_path("robot_models") + f"/{args.robot}/urdf/{args.robot}.urdf"

model = pin.buildModelFromUrdf(model_path)
data = model.createData()

print("----------------- pinocchio joint names: ")
pin_joint_names = [model.names[i] for i in range(len(model.names))]
print(pin_joint_names)


print("----------------- urdf joint names: ")
urdf_model = URDF.load(model_path)
link_names = [urdf_model.links[i].name for i in range(len(urdf_model.links))]

# Get all joint names with also fixed joints
urdf_joint_names_all = [urdf_model.joints[i].name for i in range(len(urdf_model.joints))]

urdf_joint_names = []
mimic_joint_names = []
mimic_target_joint_names = []
for joint in urdf_model.joints:
    if joint.joint_type != "fixed":
        urdf_joint_names.append(joint.name)

    if joint.mimic is not None:
        mimic_joint_names.append(joint.name)
        mimic_target_joint_names.append(joint.mimic.joint)

if 'srh' in args.robot or 'slh' in args.robot:
    print("[Warning] Using default mimic joint names for Shadow hand (Since not defined in URDF)")
    mimic_joint_names = ["rh_FFJ1", "rh_MFJ1", "rh_RFJ1", "rh_LFJ1"]
    mimic_target_joint_names = ["rh_FFJ2", "rh_MFJ2", "rh_RFJ2", "rh_LFJ2"]

print(urdf_joint_names)



if pin_joint_names[0] == "universe":
    print("[Warning] First joint in pinocchio model is universe, remove from list")
    pin_joint_names = pin_joint_names[1:]

total_joints = len(pin_joint_names)
print("--- total joints: \n", total_joints)

pin2urdf_indices = []
for joint_name in urdf_joint_names:
    index = pin_joint_names.index(joint_name)
    pin2urdf_indices.append(index)

print("--- pin2urdf_joint_indices: \n", pin2urdf_indices)

print("\n")
print("--- mimic joint names: \n", mimic_joint_names)
print("--- mimic target joint names: \n", mimic_target_joint_names)

mimic_joint_indices = []
for joint_name in mimic_joint_names:
    index = pin_joint_names.index(joint_name)
    mimic_joint_indices.append(index)

print("--- mimic_joint_indices: \n", mimic_joint_indices)

mimic_target_joint_indices = []
for joint_name in mimic_target_joint_names:
    index = pin_joint_names.index(joint_name)
    mimic_target_joint_indices.append(index)

print("--- mimic_target_joint_indices: \n", mimic_target_joint_indices)
print("\n")


pin_body_frame_names = []
for frame_id in range(model.nframes):
    frame_name = model.frames[frame_id].name  # Retrieve the frame name
    frame = model.frames[frame_id]
    if frame.type == pin.pinocchio_pywrap_default.FrameType.BODY:
        pin_body_frame_names.append(frame_name)


def find_tip_joints(input_model):
    """
    Find the tip frames of a robot model (tip frames are frames that have no child joint)
    Args:
        input_model:
    Returns:
        tip_joints: list of tip frame name
    """
    tip_joints = []
    start_id = 1 if input_model.joints[0].shortname() == "universe" else 0
    for joint_idx in range(start_id, input_model.njoints):
        joint_name = input_model.names[joint_idx]
        has_child_joint = False

        for child_joint_idx in range(start_id, input_model.njoints):
            if input_model.parents[child_joint_idx] == joint_idx:
                has_child_joint = True
                break

        if not has_child_joint:
            tip_joints.append(joint_name)

    return tip_joints

# The sequence of tip joints is consistent with the order of fingers in pinocchio model
tip_joints = find_tip_joints(model)

# Get the corresponding fingertip link names with the same order of fingers in pinocchio model
tip_frames = []
for search_joint in tip_joints:
    while True:
        urdf_joint_index = urdf_joint_names_all.index(search_joint)
        child_link = urdf_model.joints[urdf_joint_index].child

        # Iterate over all joints and check if the given link is the parent of any joint
        is_parent = False
        for joint in urdf_model.joints:
            if joint.parent == child_link:
                is_parent = True
                search_joint = joint.name
                break

        if not is_parent:
            tip_frames.append(child_link)
            break

print("--- tip_frames: \n", tip_frames)

current_path = os.path.dirname(os.path.realpath(__file__))
cfg_folder_path = os.path.join(current_path, "cfg")
if not os.path.exists(cfg_folder_path):
    os.makedirs(cfg_folder_path)

cfg_file_path = os.path.join(cfg_folder_path, f"{args.robot}.yaml")
if os.path.exists(cfg_file_path):
    user_input = input("The configuration file already exists, QUIT (q) or REWRITE (any key else) >> ")
    if user_input == "q":
        exit()
    else:
        with open(cfg_file_path, "r") as f:
            exist_content = yaml.load(f, Loader=yaml.SafeLoader)

        try:
            col_frame_pairs = exist_content["col_frame_pairs"]
        except Exception as e:
            col_frame_pairs = []
else:
    col_frame_pairs = []

if not col_frame_pairs:
    print("[Warning] Collision pairs are not defined in the configuration\n"
          "Please fill in the col_frame_pairs taking the reference to the existed link frame_names")


with open(cfg_file_path, "w") as f:
    yaml.safe_dump({
        "total_joints": total_joints,
        "pin2urdf_joint_indices": pin2urdf_indices,
        "mimic_joint_indices": mimic_joint_indices,
        "mimic_target_joint_indices": mimic_target_joint_indices,
        "tip_frames": tip_frames,
        "frame_names": pin_body_frame_names,
        "col_frame_pairs": col_frame_pairs
    },
        f, default_flow_style=None, sort_keys=False)

