import pinocchio as pin
import numpy as np
import nlopt

import rospkg
ros_package = rospkg.RosPack()

# Robot model with 1 joint rotating around the z-axis
model = pin.buildModelFromUrdf(ros_package.get_path("robot_models") + "/arm/urdf/arm.urdf")
data = model.createData()
pin.neutral(model)

frame_id = model.getFrameId('link2')


def compute_jacobian(q, frame):
    fingertip_frame_id = model.getFrameId(frame)
    jacobian = pin.computeFrameJacobian(model, data, q, fingertip_frame_id, pin.LOCAL)
    return jacobian


desired_nor = np.array([1.0, 1.0, 0])
desired_nor /= np.linalg.norm(desired_nor)

local_nor = np.array([1, 0, 0])  # x轴方向

def cost_function(q, grad):
    """
    Cost function for the optimization problem.
    Args:
        q: Joint angles.
        grad: Gradient of the cost function.
    Returns:
        cost (float): The cost value.
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    pose = data.oMf[frame_id]
    rot = pose.rotation

    world_nor = rot @ local_nor

    print(f"world_nor: {world_nor}")
    print(f"desired_nor: {desired_nor}")

    cost = 1 - np.dot(desired_nor, world_nor)

    print(f"cost: {cost}")

    if grad.size > 0:
        J = compute_jacobian(q, 'link2')

        J_pos = rot @ J[:3]
        J_rot = rot @ J[3:]

        print(f"J_pos: {J_pos}")
        print(f"J_rot: {J_rot}")

        dn_dq = np.cross(J_rot, world_nor)
        print(f"dn_dq: {dn_dq}")

        grad[:] = -np.dot(desired_nor, dn_dq)
        print(f"grad: {grad}")
    return cost


opt = nlopt.opt(nlopt.LD_LBFGS, model.nq)
opt.set_lower_bounds(np.full(model.nq, -3.14))
opt.set_upper_bounds(np.full(model.nq, 3.14))
# Set stopping criteria
opt.set_xtol_rel(1e-4)


# Set objective function
opt.set_min_objective(cost_function)

last_qpos = np.zeros(model.nq)

res = opt.optimize(last_qpos)

print(res)

