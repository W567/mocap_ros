import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import numpy as np

def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.0):
    """将裁剪框内的相机参数转换为完整图像的相机参数"""
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def process_data(out, batch):
    """处理 MANO 模型的数据"""
    right = np.array([1])

    # 提取顶点数据
    vertices = out['pred_vertices'].cpu().numpy()

    # 处理相机参数
    pred_cam = out["pred_cam"]
    pred_cam[:, 1] *= 2 * batch["right"] - 1
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()
    scaled_focal_length = 433.9375
    pred_cam_t_full = (
        cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
        .detach()
        .cpu()
        .numpy()
    )

    # 处理 3D 关键点
    pred_keypoints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()
    pred_keypoints_3d[:, :, 0] = (2 * right[:, None] - 1) * pred_keypoints_3d[:, :, 0]
    pred_keypoints_3d += pred_cam_t_full[:, None, :]  # 转换到相机坐标系
    vertices += pred_cam_t_full[:, None, :]

    # 处理全局旋转和局部旋转
    global_orient = out["pred_mano_params"]["global_orient"].squeeze(1).detach().cpu().numpy()
    joint_rot_orig = out["pred_mano_params"]["hand_pose"].squeeze(1).detach().cpu().numpy()

    # 重新映射关节顺序
    new_indices = np.array([12, 13, 14, -1, 0, 1, 2, -1, 3, 4, 5, -1, 9, 10, 11, -1, 6, 7, 8, -1])
    joint_rot = joint_rot_orig[:, new_indices, :, :]

    # 计算全局旋转矩阵
    for i, index in enumerate(new_indices):
        if index in [0, 3, 6, 9, 12]:
            joint_rot[:, i] = global_orient @ joint_rot[:, i]
        elif index == -1:
            joint_rot[:, i] = joint_rot[:, i - 1]
        else:
            joint_rot[:, i] = joint_rot[:, i - 1] @ joint_rot[:, i]

    # 转换为四元数
    rotation = global_orient[0]  # [3, 3]
    joint_rotation = joint_rot[0]  # [20, 3, 3]
    joint_rotation = np.concatenate([rotation[None, :], joint_rotation], axis=0)  # [21, 3, 3]
    joint_quat = R.from_matrix(joint_rotation).as_quat()  # [21, 4]

    return vertices, pred_keypoints_3d, joint_quat

def visualize_mano(vertices, pred_keypoints_3d, joint_quat):
    """可视化 MANO 模型和关节的局部坐标系"""
    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices[0])
    color = np.ones_like(vertices[0]) * 0.5  # 灰色
    point_cloud.colors = o3d.utility.Vector3dVector(color)

    # 创建关节的局部坐标系
    frames = []
    axes = []
    for i, joint_q in enumerate(joint_quat):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        frame.rotate(R.from_quat(joint_q).as_matrix(), center=(0,0,0))
        frame.translate(pred_keypoints_3d[0, i])
        frames.append(frame)
        
        if i in [8, 12, 16, 20]:
            axis = o3d.geometry.TriangleMesh.create_arrow(0.0005, 0.001, 0.01, 0.005)
            axis.rotate(R.from_euler('x',90,degrees=True).as_matrix(), center=(0,0,0))
            axis.rotate(R.from_quat(joint_q).as_matrix(), center=(0,0,0))
            axis.translate(pred_keypoints_3d[0, i])
            axes.append(axis)


    # 默认 Z 轴方向
    z_axis = np.array([0, 0, 1])
    # 目标方向
    target_axis = np.array([-0.55, -0.65, -0.8])
    target_axis = target_axis / np.linalg.norm(target_axis)  # 归一化
    # 计算旋转轴
    rotation_axis = np.cross(z_axis, target_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # 归一化
    # 计算旋转角度
    cos_theta = np.dot(z_axis, target_axis)
    theta = np.arccos(cos_theta)
    # 构建旋转矩阵
    rotation = R.from_rotvec(rotation_axis * theta)
    rotation_matrix = rotation.as_matrix()
    # 拇指forward方向
    thu_axis = o3d.geometry.TriangleMesh.create_arrow(0.0005, 0.001, 0.01, 0.005)
    thu_axis.rotate(rotation_matrix, center=(0,0,0))
    thu_axis.rotate(R.from_quat(joint_quat[4]).as_matrix(), center=(0,0,0))
    thu_axis.translate(pred_keypoints_3d[0, 4])


    # 可视化
    o3d.visualization.draw_geometries([point_cloud, thu_axis] + frames + axes)

# 主程序
if __name__ == "__main__":
    # 加载数据
    out = torch.load('out.pth')
    batch = torch.load('batch.pth')

    # 处理数据
    vertices, pred_keypoints_3d, joint_quat = process_data(out, batch)

    # 可视化
    visualize_mano(vertices, pred_keypoints_3d, joint_quat)
