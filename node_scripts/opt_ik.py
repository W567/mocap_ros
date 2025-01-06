#!/usr/bin/env python
import torch
import nlopt
import numpy as np
import pinocchio as pin


class OptIK:
    def __init__(self, tol=1e-4, collision_threshold=0.018, verbose=False, with_collision=True):
        self.verbose = verbose
        self.with_collision = with_collision

        self.thu_pos = np.zeros(3)
        self.thu_nor = np.zeros(3)
        self.ind_pos = np.zeros(3)
        self.ind_nor = np.zeros(3)
        self.mid_pos = np.zeros(3)
        self.mid_nor = np.zeros(3)
        self.rin_pos = np.zeros(3)
        self.rin_nor = np.zeros(3)
        self.lit_pos = np.zeros(3)
        self.lit_nor = np.zeros(3)

        # Load the robot model
        self.model = pin.buildModelFromUrdf("/home/wu/catkin_ws/src/mine/robot_models/srh_float/urdf/srh_float.urdf")
        self.data = self.model.createData()
        pin.neutral(self.model)

        # ['universe',
        #  'rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1',   # 0-3
        #  'rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1', # 4-8
        #  'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1', # 9-12
        #  'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1', # 13-16
        #  'rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1'] # 17-21
        self.names = [self.model.names[i] for i in range(len(self.model.names))]
        lower_limits = []
        upper_limits = []
        for joint_id in range(1, self.model.njoints):
            # joint_name = self.model.names[joint_id]  # Retrieve the joint name from the names list
            lower_limit = self.model.lowerPositionLimit[joint_id - 1]
            upper_limit = self.model.upperPositionLimit[joint_id - 1]
            lower_limits.append(lower_limit)
            upper_limits.append(upper_limit)

        frame_names = []
        for frame_id in range(self.model.nframes):
            frame_name = self.model.frames[frame_id].name  # Retrieve the frame name
            frame_names.append(frame_name)
        #     frame = self.model.frames[frame_id]
        #     print(f"Name: {frame.name}, Type: {frame.type}")
        # print(frame_names)

        self.total_joint_angles = 22
        self.tip_frames = ["rh_fftip", "rh_lftip", "rh_mftip", "rh_rftip", "rh_thtip"]
        self.mimic_joint_ids = np.array([3, 8, 12, 16])

        # Optimization
        self.opt = nlopt.opt(nlopt.LD_LBFGS, self.total_joint_angles)
        self.opt.set_lower_bounds(lower_limits)
        self.opt.set_upper_bounds(upper_limits)
        # Set stopping criteria
        self.opt.set_xtol_rel(tol)

        # Last(Initial) guess for joint angles (zeros or some reasonable starting guess)
        self.last_qpos = np.zeros(self.total_joint_angles)
        self.last_qpos = np.clip(self.last_qpos, lower_limits, upper_limits)

        self.pin2real = [0, 1, 2, 3,         # index
                         9, 10, 11, 12,      # middle
                        13, 14, 15, 16,      # ring
                         4, 5, 6, 7, 8,      # little
                        17, 18, 19, 20, 21]  # thumb

        if self.with_collision:
            self.col_frame_pairs = \
                [
                    # # Tip <-> Tip
                    # ("rh_fftip", "rh_mftip"),
                    # ("rh_mftip", "rh_rftip"),
                    ("rh_rftip", "rh_lftip"),
                    # Distal <-> Distal
                    ("rh_ffdistal", "rh_mfdistal"),
                    ("rh_mfdistal", "rh_rfdistal"),
                    ("rh_rfdistal", "rh_lfdistal"),
                    # # Tip <-> Distal
                    # ("rh_fftip", "rh_mfdistal"),
                    # ("rh_mftip", "rh_rfdistal"),
                    # ("rh_rftip", "rh_lfdistal"),
                    # # Distal <-> Tip
                    # ("rh_ffdistal", "rh_mftip"),
                    # ("rh_mfdistal", "rh_rftip"),
                    # ("rh_rfdistal", "rh_lftip"),
                    # Middle <-> Middle
                    ("rh_ffmiddle", "rh_mfmiddle"),
                    ("rh_mfmiddle", "rh_rfmiddle"),
                    ("rh_rfmiddle", "rh_lfmiddle"),
                    # Middle <-> Distal
                    ("rh_ffmiddle", "rh_mfdistal"),
                    ("rh_mfmiddle", "rh_rfdistal"),
                    ("rh_rfmiddle", "rh_lfdistal"),
                    # Distal <-> Middle
                    ("rh_ffdistal", "rh_mfmiddle"),
                    ("rh_mfdistal", "rh_rfmiddle"),
                    ("rh_rfdistal", "rh_lfmiddle"),

                    # lfproximal
                    ("rh_lfproximal", "rh_rfmiddle"),
                    ("rh_lfproximal", "rh_rfproximal"),

                    # Thumb
                    ("rh_thdistal", "rh_ffmiddle"),
                    ("rh_thdistal", "rh_mfmiddle"),
                    ("rh_thdistal", "rh_rfmiddle"),
                    ("rh_thdistal", "rh_lfmiddle"),
                    # ("rh_thmiddle", "rh_ffmiddle"),
                    # ("rh_thmiddle", "rh_mfmiddle"),
                    # ("rh_thmiddle", "rh_rfmiddle"),
                    # ("rh_thmiddle", "rh_lfmiddle"),
                ]

            self.col_frame_pair_indices = [(self.model.getFrameId(frame1, pin.BODY), self.model.getFrameId(frame2, pin.BODY))
                                           for frame1, frame2 in self.col_frame_pairs]
            self.col_frame_pair_indices = np.array(self.col_frame_pair_indices)

            col_frame_pair_indices = np.array(self.col_frame_pair_indices).reshape(-1)
            self.unique_col_frame_pair_indices = np.unique(col_frame_pair_indices)

            # Collision threshold
            self.collision_threshold = collision_threshold


    def forward_kinematics(self, joint_angles):
        """
        Compute the positions and orientations for all fingers given joint angles using Pinocchio.
        Args:
            joint_angles: List of joint angles for all fingers.
        """
        q = np.array(joint_angles)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)


    def get_frame_poses(self, frames):
        poses = []
        for fr in frames:
            fr_id = self.model.getFrameId(fr, pin.BODY)
            # Get the frame position (translation) and orientation (rotation matrix)
            pose = np.eye(4)
            pose[:3, :3] = self.data.oMf[fr_id].rotation
            pose[:3, 3] = self.data.oMf[fr_id].translation
            poses.append(pose)
        return poses


    def compute_jacobian(self, q,  frame):
        """
        Compute the Jacobian matrix for the fingertip given joint angles using Pinocchio.
        Args:
            q: Joint angles.
            frame: Frame name.
        Returns:
            jacobian (numpy array): The Jacobian matrix of the fingertip.
        """
        fingertip_frame_id = self.model.getFrameId(frame, pin.BODY)  # Adjust frame names

        jacobian = pin.computeFrameJacobian(self.model, self.data, q, fingertip_frame_id, pin.LOCAL_WORLD_ALIGNED)

        return jacobian

    def objective_function(self, desired_positions, desired_normals, fingertip_frames, last_qpos):

        desired_positions = torch.as_tensor(desired_positions, dtype=torch.float32)
        desired_positions.requires_grad_(False)
        desired_normals = torch.as_tensor(desired_normals, dtype=torch.float32)
        desired_normals.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos = x.copy()
            # Set the mimic joints to the previous joint angles
            qpos[self.mimic_joint_ids] = qpos[self.mimic_joint_ids - 1]
            self.forward_kinematics(qpos)
            actual_tip_poses = self.get_frame_poses(fingertip_frames)
            tip_body_pos = np.array([pose[:3, 3] for pose in actual_tip_poses])

            torch_tip_body_pos = torch.as_tensor(tip_body_pos)
            torch_tip_body_pos.requires_grad_()

            pos_dist = torch.norm(torch_tip_body_pos - desired_positions, dim=1, keepdim=False).sum()

            ## Collision checking ======================================================================================
            col_cost = None
            col_indices = None
            torch_frame_rela_position = None
            if self.with_collision:
                # Frame positions for collision checking
                frame_positions = {frame_id: self.data.oMf[int(frame_id)].translation
                                   for frame_id in self.unique_col_frame_pair_indices}
                torch_frame_positions_1 = torch.as_tensor(
                    np.array([frame_positions[frame_id_1] for frame_id_1, _ in self.col_frame_pair_indices]))
                torch_frame_positions_2 = torch.as_tensor(
                    np.array([frame_positions[frame_id_2] for _, frame_id_2 in self.col_frame_pair_indices]))
                torch_frame_rela_position = torch_frame_positions_1 - torch_frame_positions_2
                torch_frame_rela_position.requires_grad_()

                # Distances between specified frame pairs
                distances = torch.norm(torch_frame_rela_position, dim=1, keepdim=False)
                # Find the indices of the pairs that are closer than the threshold (collision detected)
                col_indices = torch.where(distances < self.collision_threshold)[0]

                all_col_cost = self.collision_threshold - distances[col_indices]
                col_cost = all_col_cost.sum()

                if self.verbose:
                    for col_index in col_indices:
                        print(f"{self.col_frame_pairs[col_index]} Potential collision detected!"
                              f"Distance: {distances[col_index]}")
            ## Collision checking ======================================================================================

                objective_result = pos_dist.cpu().detach().item() + col_cost.cpu().detach().item()
            else:
                objective_result = pos_dist.cpu().detach().item()

            if grad.size > 0:
                jacobians = []

                for frame in fingertip_frames:
                    link_kinematics_jacobian = self.compute_jacobian(qpos, frame)[:3, ...]
                    jacobians.append(link_kinematics_jacobian)

                jacobians = np.stack(jacobians, axis=0)
                pos_dist.backward()
                grad_pos = torch_tip_body_pos.grad.cpu().numpy()[:, None, :]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                # Regularize the joint angles to the previous joint angles (smoothness term)
                grad_qpos += 2 * 4e-3 * (x - last_qpos)

                if self.with_collision:
                    # Calculate the gradient for the collision cost only if there are collisions
                    if len(col_indices) > 0:
                        frame_jacobians = []
                        for frame_id_1, frame_id_2 in self.col_frame_pair_indices[col_indices.numpy()]:
                            jacobian_1 = self.compute_jacobian(qpos, self.model.frames[int(frame_id_1)].name)[:3, ...]
                            jacobian_2 = self.compute_jacobian(qpos, self.model.frames[int(frame_id_2)].name)[:3, ...]
                            frame_jacobians.append(jacobian_1 - jacobian_2)

                        frame_jacobians = np.stack(frame_jacobians, axis=0)
                        col_cost.backward()
                        grad_frame = torch_frame_rela_position.grad.cpu().numpy()[col_indices, None, :]

                        grad_frame_qpos = np.matmul(grad_frame, frame_jacobians)
                        grad_frame_qpos = grad_frame_qpos.mean(1).sum(0)

                        grad_qpos += grad_frame_qpos

                for mimic_id in self.mimic_joint_ids:
                    grad_qpos[mimic_id] = 0  # Mimic joints don’t contribute to the optimization gradient

                grad[:] = grad_qpos[:]

            return objective_result

        return objective

    def set_target(self, idx=0):
        # open hand
        if idx == 0:
            self.ind_pos = np.array([0.033, 0.0, 0.191])
            self.ind_nor = np.array([0.0, -1.0, 0.0])
            self.lit_pos = np.array([-0.033, 0.0, 0.1826])
            self.lit_nor = np.array([0.0, -1.0, 0.0])
            self.mid_pos = np.array([0.011, 0.0, 0.195])
            self.mid_nor = np.array([0.0, -1.0, 0.0])
            self.rin_pos = np.array([-0.011, 0.0, 0.191])
            self.rin_nor = np.array([0.0, -1.0, 0.0])
            self.thu_pos = np.array([0.10294291, -0.00858, 0.09794291])
            self.thu_nor = np.array([-0.70710678, 0.0, 0.70710678])
        # envelop
        elif idx == 1:
            self.ind_pos = np.array([0.04012684, -0.04272182, 0.15803178])
            self.ind_nor = np.array([-0.30107772, -0.43725329, -0.84744425])
            self.mid_pos = np.array([0.00327784, -0.05243527, 0.151299])
            self.mid_nor = np.array([0.12893927, -0.02844863, -0.99124434])
            self.rin_pos = np.array([-0.02002411, -0.05031957, 0.14441163])
            self.rin_nor = np.array([0.35844603, -0.06347011, -0.93139035])
            self.lit_pos = np.array([-0.04417813, -0.04551897, 0.1227557])
            self.lit_nor = np.array([0.33163346, -0.41941064, -0.84505264])
            self.thu_pos = np.array([0.04887634, -0.07812261, 0.08955634])
            self.thu_nor = np.array([-0.73207116, -0.54427482, -0.40967881])
        # collision
        elif idx == 2:
            self.ind_pos = np.array([0.024, -0.058, 0.169])
            self.ind_nor = np.array([-0.30107772, -0.43725329, -0.84744425])
            self.mid_pos = np.array([0.019, -0.055, 0.168])
            self.mid_nor = np.array([0.12893927, -0.02844863, -0.99124434])
            self.rin_pos = np.array([-0.02002411, -0.05031957, 0.14441163])
            self.rin_nor = np.array([0.35844603, -0.06347011, -0.93139035])
            self.lit_pos = np.array([-0.04417813, -0.04551897, 0.1227557])
            self.lit_nor = np.array([0.33163346, -0.41941064, -0.84505264])
            self.thu_pos = np.array([0.04887634, -0.07812261, 0.08955634])
            self.thu_nor = np.array([-0.73207116, -0.54427482, -0.40967881])
        else:
            raise ValueError("Invalid target index")


    def optimize(self):
        desired_positions = np.array([self.ind_pos, self.lit_pos, self.mid_pos, self.rin_pos, self.thu_pos])
        desired_normals = np.array([self.ind_nor, self.lit_nor, self.mid_nor, self.rin_nor, self.thu_nor])

        # Set objective function
        self.opt.set_min_objective(
            self.objective_function(desired_positions, desired_normals, self.tip_frames, self.last_qpos))

        try:
            res = self.opt.optimize(self.last_qpos)
            # Set the mimic joints to the previous joint angles
            res[self.mimic_joint_ids] = res[self.mimic_joint_ids - 1]
            self.last_qpos = res

            return res[self.pin2real]
        except Exception as e:
            if self.verbose:
                print(f"Optimization failed: {e}")
            return self.last_qpos[self.pin2real]


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Use index = 1, envelop condition")
        pose_index = 1
    else:
        pose_index = sys.argv[1]
    handle = OptIK()
    handle.set_target(pose_index)
    result = handle.optimize()
    print("====== result: \n", np.array2string(result, separator=', '))
