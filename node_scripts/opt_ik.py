#!/usr/bin/env python
import os
import yaml
import torch
import nlopt
import numpy as np
import pinocchio as pin

import rospkg
ros_package = rospkg.RosPack()


class OptIK:
    def __init__(self,
                 robot="srh_float",
                 tol=1e-4,
                 nor_weight=0.01,
                 col_weight=1.0,
                 collision_threshold=0.018,
                 verbose=False,
                 with_collision=True
                 ):

        self.verbose = verbose
        self.with_collision = with_collision
        self.nor_weight = nor_weight
        self.col_weight = col_weight
        self.collision_threshold = collision_threshold

        # Load the robot model
        self.model = pin.buildModelFromUrdf(ros_package.get_path("robot_models") + f"/{robot}/urdf/{robot}.urdf")
        self.data = self.model.createData()

        current_path = os.path.dirname(os.path.realpath(__file__))
        cfg_file_path = os.path.join(current_path, "cfg", f"{robot}.yaml")
        with open(cfg_file_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
            self.total_joints = cfg["total_joints"]
            self.pin2urdf_joint_indices = cfg["pin2urdf_joint_indices"]
            self.tip_frames = cfg["tip_frames"]
            self.mimic_joint_indices = np.array(cfg["mimic_joint_indices"])
            self.mimic_target_joint_indices = np.array(cfg["mimic_target_joint_indices"])
            self.col_frame_pairs = cfg["col_frame_pairs"]
            self.tip_normals = np.array(cfg["tip_normals"])[..., np.newaxis]

        num_fingers = len(self.tip_frames)
        assert num_fingers <= 5, "Too many fingers, only support up to 5 fingers"
        finger_name_prefixes = ["thu", "ind", "mid", "rin", "lit"]
        for finger in finger_name_prefixes[:num_fingers]:
            setattr(self, f"{finger}_pos", np.zeros(3))
            setattr(self, f"{finger}_nor", np.zeros(3))

        lower_limits = []
        upper_limits = []
        for joint_id in range(self.total_joints):
            lower_limit = self.model.lowerPositionLimit[joint_id]
            upper_limit = self.model.upperPositionLimit[joint_id]
            lower_limits.append(lower_limit)
            upper_limits.append(upper_limit)

        # Optimization
        self.opt = nlopt.opt(nlopt.LD_LBFGS, self.total_joints)
        self.opt.set_lower_bounds(lower_limits)
        self.opt.set_upper_bounds(upper_limits)
        # Set stopping criteria
        self.opt.set_xtol_rel(tol)

        # Last(Initial) guess for joint angles (zeros or some reasonable starting guess)
        self.last_qpos = np.zeros(self.total_joints)
        self.last_qpos = np.clip(self.last_qpos, lower_limits, upper_limits)

        if self.with_collision:
            self.col_frame_pairs = np.array(self.col_frame_pairs)
            if len(self.col_frame_pairs) > 0:
                assert self.col_frame_pairs.shape[1] == 2, "Invalid collision frame pairs shape, should be (n, 2)"
            self.col_frame_pair_indices = [(self.model.getFrameId(frame1, pin.BODY), self.model.getFrameId(frame2, pin.BODY))
                                           for frame1, frame2 in self.col_frame_pairs]
            self.col_frame_pair_indices = np.array(self.col_frame_pair_indices)

            col_frame_pair_indices = np.array(self.col_frame_pair_indices).reshape(-1)
            self.unique_col_frame_pair_indices = np.unique(col_frame_pair_indices)


    def forward_kinematics(self, joint_angles):
        """
        Compute the positions and orientations for all fingers given joint angles using Pinocchio.
        Args:
            joint_angles: List of joint angles for all fingers.
        """
        q = np.array(joint_angles)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)


    def get_frame_pose(self, frame):
        frame_id = self.model.getFrameId(frame, pin.BODY)
        pose = np.eye(4)
        pose[:3, :3] = self.data.oMf[frame_id].rotation
        pose[:3, 3] = self.data.oMf[frame_id].translation
        return pose


    def get_frame_poses(self, frames):
        poses = []
        for fr in frames:
            pose = self.get_frame_pose(fr)
            poses.append(pose)
        return np.array(poses)


    def compute_jacobian(self, q,  frame):
        """
        Compute the Jacobian matrix for the fingertip given joint angles using Pinocchio.

            The LOCAL_WORLD_ALIGNED frame convention corresponds to the frame centered
            on the moving part (Joint, Frame, etc.) but with axes aligned with the frame of the Universe.
            This a MIXED representation between the LOCAL and the WORLD conventions.
            https://docs.ros.org/en/kinetic/api/pinocchio/html/group__pinocchio__multibody.html

        Args:
            q: Joint angles.
            frame: Frame name.
        Returns:
            jacobian (numpy array): The Jacobian matrix of the fingertip.
        """
        frame_id = self.model.getFrameId(frame, pin.BODY)  # Adjust frame names
        jacobian = pin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
        return jacobian


    def objective_function(self, desired_positions, desired_normals, fingertip_frames, last_qpos):
        desired_positions = torch.as_tensor(desired_positions, dtype=torch.float32)
        desired_positions.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos = x.copy()
            # Set the mimic joints to the previous joint angles
            qpos[self.mimic_joint_indices] = qpos[self.mimic_target_joint_indices]
            self.forward_kinematics(qpos)

            ## Position distance =======================================================================================
            actual_tip_poses = self.get_frame_poses(fingertip_frames)
            tip_body_pos = actual_tip_poses[:, :3, 3]
            torch_tip_body_pos = torch.as_tensor(tip_body_pos)
            torch_tip_body_pos.requires_grad_()
            pos_dist = torch.norm(torch_tip_body_pos - desired_positions, dim=1, keepdim=False).sum()
            ## Position distance =======================================================================================

            ## Normal distance =========================================================================================
            tip_body_nor = (actual_tip_poses[:, :3, :3] @ self.tip_normals).squeeze(-1)
            torch_tip_body_nor = torch.as_tensor(tip_body_nor)
            nor_dist = (1 - torch.sum(torch_tip_body_nor * desired_normals, dim=1)).sum()
            pos_dist += nor_dist * self.nor_weight
            ## Normal distance =========================================================================================

            ## Collision Cost ==========================================================================================
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
                objective_result = pos_dist.cpu().detach().item() + col_cost.cpu().detach().item() * self.col_weight
            else:
                objective_result = pos_dist.cpu().detach().item()
            ## Collision Cost ==========================================================================================

            if grad.size > 0:
                jacobians = []
                jacobians_omega = []
                for i, frame in enumerate(fingertip_frames):
                    link_kinematics_jacobian = self.compute_jacobian(qpos, frame)
                    jacobians.append(link_kinematics_jacobian[:3, ...])
                    jacobians_omega.append(link_kinematics_jacobian[3:, ...])

                ## Position gradient ===================================================================================
                jacobians = np.stack(jacobians, axis=0)
                pos_dist.backward()
                grad_pos = torch_tip_body_pos.grad.cpu().numpy()[:, None, :]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)
                ## Position gradient ===================================================================================

                ## Normal gradient =====================================================================================
                jacobians_omega = np.stack(jacobians_omega, axis=0)
                tip_body_nor_tmp = tip_body_nor[:, :, np.newaxis]
                tip_body_nor_tmp = np.repeat(tip_body_nor_tmp, 22, axis=2)

                dn_dq = np.cross(jacobians_omega, tip_body_nor_tmp, axis=1)

                grad_nor_qpos = -dn_dq * desired_normals[:, :, np.newaxis]
                grad_nor_qpos = grad_nor_qpos.sum(1).mean(0)
                grad_qpos += grad_nor_qpos * self.nor_weight
                ## Normal gradient =====================================================================================

                ## Regularize the joint angles to the previous joint angles (smoothness term) ==========================
                # TODO (Wu): weight should be adjusted
                grad_qpos += 2 * 4e-3 * (x - last_qpos)
                ## Regularize the joint angles to the previous joint angles (smoothness term) ==========================

                ## Calculate the gradient for the collision cost =======================================================
                if self.with_collision: # only if there are collisions
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

                        grad_qpos += grad_frame_qpos * self.col_weight
                ## Calculate the gradient for the collision cost =======================================================

                for mimic_id in self.mimic_joint_indices:
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
            res[self.mimic_joint_indices] = res[self.mimic_target_joint_indices]
            self.last_qpos = res

            return res[self.pin2urdf_joint_indices]
        except Exception as e:
            if self.verbose:
                print(f"Optimization failed: {e}")
            return self.last_qpos[self.pin2urdf_joint_indices]


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
