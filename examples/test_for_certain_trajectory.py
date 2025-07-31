"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""

import time
import os
import joblib
import torch
import viser
import numpy as np

from yourdfpy import URDF
from scipy.spatial.transform import Rotation as R
from torch import nn
from viser.extras import ViserUrdf

import pyroki_snippets as pks
import pyroki as pk
from NNmodel import ElbowNNWrapper

def load_motion_from_pkl(pkl_path: str, motion_key: str = None):
    """
    加载指定的 hand_elbow_trajectory_test.pkl 中的某一条轨迹数据。

    参数:
        pkl_path (str): pkl 文件路径
        motion_key (str): 可选，指定要加载的轨迹 key

    返回:
        motion_data (dict): 指定轨迹的数据
        all_keys (list): 所有可用轨迹的 key（用于选择）
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"{pkl_path} not found!")

    data = joblib.load(pkl_path)
    print(f"✅ 成功载入 {pkl_path}")
    print(f"📦 动作数量: {len(data)}\n")

    all_keys = list(data.keys())

    if motion_key is None:
        print("📌 可选 motion_key 列表：")
        for i, k in enumerate(all_keys):
            print(f"  [{i}] {k}")
        print("\n❗未指定 motion_key，请参考上方列表传入 key 使用。")
        return None, all_keys

    if motion_key not in data:
        raise ValueError(f"❌ motion_key='{motion_key}' 不在数据中。可选项为：{all_keys}")

    motion_data = data[motion_key]
    print(f"🎯 已选择 motion: {motion_key}")
    print(f"📐 包含字段: {list(motion_data.keys())}")

    return motion_data, all_keys

def extract_tcp_elbow_from_motion(motion_data: dict, to_shoulder=True):
    """
    提取指定 motion_data 中的左右手 TCP 和肘部信息（位置 + 旋转），按帧组织。

    参数:
        motion_data (dict): 从 load_motion_from_pkl 得到的 motion_data
        use_local (bool): 如果为 True，则使用以肩关节为参考的局部坐标；否则使用世界坐标

    返回:
        result (dict): 包含左右手 TCP 和 elbow 的轨迹，按帧排列
    """
    left_tcp_traj, right_tcp_traj = [], []
    left_elbow_traj, right_elbow_traj = [], []

    for frame in motion_data["frames"]:
        if to_shoulder:
            left_tcp = {
                "pos": frame["left_shoulder"]["tcp_pos"],
                "rot": frame["left_shoulder"]["tcp_rot"]
            }
            right_tcp = {
                "pos": frame["right_shoulder"]["tcp_pos"],
                "rot": frame["right_shoulder"]["tcp_rot"]
            }
            left_elbow = {
                "pos": frame["left_shoulder"]["elbow_pos"],
                "rot": frame["left_shoulder"]["elbow_rot"]
            }
            right_elbow = {
                "pos": frame["right_shoulder"]["elbow_pos"],
                "rot": frame["right_shoulder"]["elbow_rot"]
            }
        else:
            left_tcp = {
                "pos": frame["left_root"]['tcp_pos'],
                "rot": frame["left_root"]["tcp_rot"]
            }
            right_tcp = {
                "pos": frame["right_root"]['tcp_pos'],
                "rot": frame["right_root"]["tcp_rot"]
            }
            left_elbow = {
                "pos": frame["left_root"]['elbow_pos'],
                "rot": frame["left_root"]["elbow_rot"]
            }
            right_elbow = {
                "pos": frame["right_root"]['elbow_pos'],
                "rot": frame["right_root"]["elbow_rot"]
            }

        left_tcp_traj.append(left_tcp)
        right_tcp_traj.append(right_tcp)
        left_elbow_traj.append(left_elbow)
        right_elbow_traj.append(right_elbow)

    return {
        "left_tcp": left_tcp_traj,
        "right_tcp": right_tcp_traj,
        "left_elbow": left_elbow_traj,
        "right_elbow": right_elbow_traj
    }

def extract_tcp_elbow_from_amass_motion(motion_data: dict, to_shoulder=False):
    """
    motion_data["frame"] 是 dict，每个 key 对应 shape=(N, 3 or 4) 的 np.array
    """
    frames = motion_data["frame"]
    N = len(frames['left_hand_root_pos'])  # 帧数

    left_tcp_traj, right_tcp_traj = [], []
    left_elbow_traj, right_elbow_traj = [], []

    for i in range(N):
        if to_shoulder:
            left_tcp = {
                "pos": frames['left_hand_shoulder_pos'][i],
                "rot": frames["left_hand_shoulder_rot"][i]
            }
            right_tcp = {
                "pos": frames['right_hand_shoulder_pos'][i],
                "rot": frames["right_hand_shoulder_rot"][i]
            }
            left_elbow = {
                "pos": frames['left_elbow_shoulder_pos'][i],
                "rot": frames["left_elbow_shoulder_rot"][i]
            }
            right_elbow = {
                "pos": frames['right_elbow_shoulder_pos'][i],
                "rot": frames["right_elbow_shoulder_rot"][i]
            }
        else:
            left_tcp = {
                "pos": frames['left_hand_root_pos'][i],
                "rot": frames["left_hand_root_rot"][i]
            }
            right_tcp = {
                "pos": frames['right_hand_root_pos'][i],
                "rot": frames["right_hand_root_rot"][i]
            }
            left_elbow = {
                "pos": frames['left_elbow_root_pos'][i],
                "rot": frames["left_elbow_root_rot"][i]
            }
            right_elbow = {
                "pos": frames['right_elbow_root_pos'][i],
                "rot": frames["right_elbow_root_rot"][i]
            }

        left_tcp_traj.append(left_tcp)
        right_tcp_traj.append(right_tcp)
        left_elbow_traj.append(left_elbow)
        right_elbow_traj.append(right_elbow)

    return {
        "left_tcp": left_tcp_traj,
        "right_tcp": right_tcp_traj,
        "left_elbow": left_elbow_traj,
        "right_elbow": right_elbow_traj
    }

def root_to_shoulder_pose(pose_root, shoulder_pose_root):
    """
    将一个姿态 (pos, quat) 从 world 坐标系变换到 shoulder 局部坐标系。
    输入:
        pose_root: (pos, quat_xyzw)
        shoulder_pose_root: (pos, quat_xyzw)
    输出:
        (pos_local, quat_local): 相对于 shoulder 的局部位姿
    """
    pos_root, quat_root = pose_root
    shoulder_pos, shoulder_quat = shoulder_pose_root

    R_world = R.from_quat(quat_root)
    R_shoulder = R.from_quat(shoulder_quat)

    # 坐标变换：世界坐标系 -> 肩膀局部坐标系
    pos_local = R_shoulder.inv().apply(pos_root - shoulder_pos)
    R_local = R_shoulder.inv() * R_world
    quat_local = R_local.as_quat()

    return pos_local, quat_local

def root_to_shoulder(data_28, shoulder_pose_world):
    """
    输入一个 (28,) 的向量，分别是:
        left_hand_pos(3) + quat(4)
        right_hand_pos(3) + quat(4)
        left_elbow_pos(3) + quat(4)
        right_elbow_pos(3) + quat(4)

    输出：全部转换为肩膀坐标系下的同样格式 (28,) 向量
    """
    out = []

    for i in range(0, 28, 7):  # 每7个为一组
        pos = data_28[i:i+3]
        quat = data_28[i+3:i+7]
        pos_local, quat_local = root_to_shoulder_pose((pos, quat), shoulder_pose_world)
        out += list(pos_local) + list(quat_local)

    return np.array(out)

def shoulder_to_root(pose_local, shoulder_pose_world):
    """
    将 shoulder 坐标系下的 pose 映射到 root 坐标系下。

    参数:
        pose_local: (pos, quat)，在 shoulder 坐标系下的位姿
        shoulder_pose_world: (pos, quat)，shoulder 的 root 位姿

    返回:
        (pos_world, quat_world): 在 world 坐标系下的位姿
    """
    pos_local, quat_local = pose_local
    shoulder_pos, shoulder_quat = shoulder_pose_world

    shoulder_rot = R.from_quat(shoulder_quat)

    # 位置变换
    pos_world = shoulder_rot.apply(pos_local) + shoulder_pos

    # 旋转变换
    rot_local = R.from_quat(quat_local)
    quat_world = (shoulder_rot * rot_local).as_quat()

    return pos_world, quat_world

def get_link_pose(robot, q, link_name):
    """
    获取 robot 在状态 q 下，link_name 的世界坐标系下位姿 (pos, quat_xyzw)
    """
    idx = robot.links.names.index(link_name)
    pose = robot.forward_kinematics(q[None])[0, idx]
    quat_xyzw = np.roll(pose[:4], -1)  # wxyz -> xyzw
    pos_xyz = pose[4:]
    return pos_xyz, quat_xyzw


def main():
    """Main function for basic IK."""
    use_model = True
    urdf_path = "/home/wzh-2004/3DPOSE_TEST/human2humanoid/resources/robots/h1_2/urdf/h1_2.urdf"
    mesh_dir = "/home/wzh-2004/3DPOSE_TEST/human2humanoid/resources/robots/h1_2/meshes"
    urdf = URDF.load(
        urdf_path,
        mesh_dir=mesh_dir,                 # 用于解析 mesh 文件路径
        load_meshes=True,                 # 如果你想渲染机器人模型
        build_scene_graph=True           # 如果你需要进行可视化或碰撞检测
    )
    target_link_name = "L_hand_base_link"
    target_elbow_link_name = "left_elbow_pitch_link"
    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # 读取默认关节配置
    default_cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    # 计算所有 link 的初始位姿（SE(3): wxyz + xyz）
    inital_link_poses = robot.forward_kinematics(default_cfg[None])  # shape: (1, link_count, 7)

    # 取 batch 的第一个结果
    inital_link_poses = inital_link_poses[0]  # shape: (link_count, 7)
    inital_left_tcp_pose = inital_link_poses[robot.links.names.index(target_link_name)]
    inital_left_elbow_pose = inital_link_poses[robot.links.names.index(target_elbow_link_name)]
    inital_right_tcp_pose = inital_link_poses[robot.links.names.index("R_hand_base_link")]
    inital_right_elbow_pose = inital_link_poses[robot.links.names.index("right_elbow_pitch_link")]

    shoulder_pos, shoulder_quat = get_link_pose(robot, default_cfg, "left_shoulder_pitch_link")
    solution_previous = None
    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    motion_data, _ = load_motion_from_pkl(
    "/home/wzh-2004/3DPOSE_TEST/human2humanoid/data/new_robot/amass_all.pkl",
    motion_key="0-ACCAD_Female1General_c3d_A6- lift box t2_poses"
    )

    # 提取轨迹
    # traj_local = extract_tcp_elbow_from_motion(motion_data, to_shoulder=True)
    traj_root = extract_tcp_elbow_from_amass_motion(motion_data, to_shoulder=False)
    tcp_traj = traj_root['left_tcp']
    elbow_traj = traj_root['left_elbow']

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    rot_mat_z_neg90 = R.from_euler("z", -90, degrees=True).as_matrix()
    quat_correction = R.from_euler("z", -90, degrees=True).as_quat()

    if use_model:
        model_path = "/home/wzh-2004/3DPOSE_TEST/human2humanoid/elbow_mlp_model.pth"
        model = ElbowNNWrapper()
        model.load(model_path)
        # 用于存储当前帧和前四帧的tcp_pose和 elbow_pose
        '''
        存储结构：
        输入格式:
            前面 5 帧的每一帧数据格式为：
                frames["left_hand_shoulder_pos"][idx],       # (3,)
                frames["left_hand_shoulder_rot"][idx],       # (4,)
                frames["right_hand_shoulder_pos"][idx],      # (3,)
                frames["right_hand_shoulder_rot"][idx],      # (4,)
                frames["left_elbow_shoulder_pos"][idx],      # (3,)
                frames["left_elbow_shoulder_rot"][idx],      # (4,)
                frames["right_elbow_shoulder_pos"][idx],     # (3,)
                frames["right_elbow_shoulder_rot"][idx],     # (4,)
            当前帧 t 时刻的目标 tcp 为：
                frames["left_hand_shoulder_pos"][next_idx],     # (3,)
                frames["left_hand_shoulder_rot"][next_idx],     # (4,)
                frames["right_hand_shoulder_pos"][next_idx],    # (3,)
                frames["right_hand_shoulder_rot"][next_idx],    # (4,)
        输出格式：
            当前帧 t 时刻的目标 elbow 的位置：
                frames["left_elbow_shoulder_pos"][next_idx],    # (3,)
                frames["left_elbow_shoulder_rot"][next_idx],    # (4,)
                frames["right_elbow_shoulder_pos"][next_idx],   # (3,)
                frames["right_elbow_shoulder_rot"][next_idx],   # (4,)
        '''
        recent_tcp_elbow_data_root = [ ]
        initial_frame = np.concatenate([
            inital_left_tcp_pose[:3],  # pos
            inital_left_tcp_pose[3:],  # rot
            inital_right_tcp_pose[:3],  # pos
            inital_right_tcp_pose[3:],  # rot
            inital_left_elbow_pose[:3],  # pos
            inital_left_tcp_pose[3:],  # rot
            inital_right_elbow_pose[:3],  # pos
            inital_right_tcp_pose[3:],  # rot
        ])
        for _ in range(5):
            recent_tcp_elbow_data_root.append(initial_frame)

    while True:
        # 循环遍历 tcp_traj，将当前帧 t 时刻的 tcp_pos 作为 IK 目标
        for tcp_frame, elbow_frame in zip(tcp_traj, elbow_traj):
            start_time = time.time()
            target_left_tcp_position_root = np.array(tcp_frame["pos"])
            target_left_tcp_xyzw_root = np.array(tcp_frame["rot"])
            target_left_elbow_position_root = np.array(elbow_frame["pos"])
            target_left_elbow_rot_quat = np.array(elbow_frame["rot"])

            # 添加绕 Z 轴 -90° 的旋转修正
            _target_left_tcp_position_root = rot_mat_z_neg90 @ target_left_tcp_position_root
            _corrected_xyzw_root = R.from_quat(target_left_tcp_xyzw_root)
            _target_left_tcp_wxyz_root = np.roll(_corrected_xyzw_root.as_quat(), 1) # 转为 wxyz

            _target_left_elbow_position_root = rot_mat_z_neg90 @ target_left_elbow_position_root
            _corrected_elbow_xyzw_root = R.from_quat(target_left_elbow_rot_quat)
            _target_left_elbow_rot_quat_root = np.roll(_corrected_elbow_xyzw_root.as_quat(), 1) # 转为 wxyz

            if use_model:
                # 存储最近的目标
                recent_target_data_root = np.concatenate([
                    target_left_tcp_position_root,
                    target_left_tcp_xyzw_root,
                    inital_right_tcp_pose[:3],  # pos
                    inital_right_tcp_pose[3:],  # rot
                ])

                # 将需要传入到model中的 pose 队列，坐标系转换为 shoulder 局部坐标系
                x_seq_data = [root_to_shoulder(x, (shoulder_pos,shoulder_quat))
                              for x in recent_tcp_elbow_data_root]
                x_nex_tcp = root_to_shoulder(
                    np.concatenate([
                        recent_target_data_root,                 # 14维
                        np.zeros(3),           # pos
                        np.array([0, 0, 0, 1]),  # unit quaternion
                        np.zeros(3),           # pos
                        np.array([0, 0, 0, 1])   # unit quaternion
                    ]),
                    (shoulder_pos,shoulder_quat)
                )

                with torch.no_grad():
                    # 将需要传入到model中的 pose 队列，转换为 torch tensor
                    x_seq = torch.from_numpy(np.array(x_seq_data)).float().to('cuda')  # (5, 28)
                    x_nex_tcp = torch.from_numpy(np.array(x_nex_tcp[:14])).float().to('cuda')  # (1, 14)
                    input_tensor = torch.cat((x_seq.flatten().unsqueeze(0), x_nex_tcp.unsqueeze(0)), dim=1)  # shape: (1, 154)

                    # 预测肘部pos和root，表示在shoulder坐标系下
                    _predictions = model.predict(input_tensor)
                    _pred_target_left_elbow_position_shoulder = _predictions[0, :3].cpu().numpy()
                    _pred_target_left_elbow_rot_quat_shoulder = _predictions[0, 3:7].cpu().numpy()
                    _pred_target_right_elbow_position_shoulder = _predictions[0, 7:10].cpu().numpy()
                    _pred_target_right_elbow_rot_quat_shoulder = _predictions[0, 10:14].cpu().numpy()

                    # 将预测的 elbow pose 从 shoulder 坐标系转换到 root 坐标系
                    pred_left_elbow_pose_root = shoulder_to_root(
                        (_pred_target_left_elbow_position_shoulder, _pred_target_left_elbow_rot_quat_shoulder),
                        (shoulder_pos, shoulder_quat)
                    )
                    pred_right_elbow_pose_root = shoulder_to_root(
                        (_pred_target_right_elbow_position_shoulder, _pred_target_right_elbow_rot_quat_shoulder),
                        (shoulder_pos, shoulder_quat)
                    )
                    # 将预测的 root 坐标系下 elbow pose 分解为位置和旋转四元数
                    _pred_target_left_elbow_position_root, _pred_target_left_elbow_rot_quat_root = pred_left_elbow_pose_root
                    _pred_target_right_elbow_position_root, _pred_target_right_elbow_rot_quat_root = pred_right_elbow_pose_root

                    # 将预测 root 坐标系下的结果加入 recent_target_data_root
                    recent_target_data_root = np.concatenate([
                        recent_target_data_root,
                        _pred_target_left_elbow_position_root,
                        _pred_target_left_elbow_rot_quat_root,
                        _pred_target_right_elbow_position_root,
                        _pred_target_right_elbow_rot_quat_root,
                    ])

                    # 将 recent_target_data_root 加入 recent_tcp_elbow_data_root（保持队列中只有五帧的数据）
                    recent_tcp_elbow_data_root.append(recent_target_data_root)
                    if len(recent_tcp_elbow_data_root) > 5:
                        recent_tcp_elbow_data_root.pop(0)  # 保持存储最近5帧

                solution = pks.solve_ik(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_position=_target_left_tcp_position_root,
                    target_wxyz=_target_left_tcp_wxyz_root,
                    target_elbow_position=_pred_target_left_elbow_position_root,
                    target_elbow_rot_quat=_pred_target_left_elbow_rot_quat_root,
                    target_elbow_link_name=target_elbow_link_name,
                    init_q=solution_previous if solution_previous is not None else default_cfg,
                )

            else:
                # 直接使用手动指定的目标位置和旋转
                solution = pks.solve_ik(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_position=_target_left_tcp_position_root,
                    target_wxyz=_target_left_tcp_wxyz_root,
                    target_elbow_position=_target_left_elbow_position_root,
                    target_elbow_rot_quat=_target_left_elbow_rot_quat_root,
                    target_elbow_link_name=target_elbow_link_name,
                    init_q=solution_previous if solution_previous is not None else default_cfg,
                )

            # 更新shoulder的位姿
            shoulder_pos, shoulder_quat = get_link_pose(robot, solution, "left_shoulder_pitch_link")
            # 更新机器人状态
            solution_previous = solution
            # print(f"target:{_target_left_tcp_position_root}")
            server.scene.add_frame(
                f"/{'ik_target'}",
                position=_target_left_tcp_position_root,
                wxyz=_target_left_tcp_wxyz_root,
                axes_length=0.2,
                axes_radius=0.01,
            )

            # Update timing handle.
            elapsed_time = time.time() - start_time
            timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

            # Update visualizer.
            urdf_vis.update_cfg(solution)

            # 可选：每帧暂停一下，便于观察
            time.sleep(0.1)


if __name__ == "__main__":
    main()
