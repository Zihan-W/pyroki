"""Trajectory Optimization

Basic Trajectory Optimization using PyRoKi.

Robot going over a wall, while avoiding world-collisions.
"""

import os
import time
import joblib
from typing import Literal
from yourdfpy import URDF

import numpy as np
import pyroki as pk
import trimesh
import tyro
import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description

import pyroki_snippets as pks

from scipy.spatial.transform import Rotation as R

# Z轴逆时针90°，等价于将原始的 root_link 坐标系转回 world
z_fix = R.from_euler('z', -90, degrees=True)

def correct_to_world(quat_wxyz, pos=None):
    """将 TCP 相对于 root_link 的姿态转换为相对于 world 的姿态"""
    # 四元数：wxyz → xyzw
    r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    r_world = z_fix * r
    quat_corrected = r_world.as_quat()
    quat_wxyz_corrected = np.array([quat_corrected[3], quat_corrected[0], quat_corrected[1], quat_corrected[2]])

    if pos is None:
        return quat_wxyz_corrected
    else:
        pos_corrected = z_fix.apply(pos)
        return quat_wxyz_corrected, pos_corrected

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
            raise NotImplementedError("当前frame结构下不支持shoulder系，除非补充数据")
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

urdf_path = "/home/wzh-2004/3DPOSE_TEST/human2humanoid/resources/robots/h1_2/urdf/h1_2.urdf"
mesh_dir = "/home/wzh-2004/3DPOSE_TEST/human2humanoid/resources/robots/h1_2/meshes"
urdf = URDF.load(
    urdf_path,
    mesh_dir=mesh_dir,                 # 用于解析 mesh 文件路径
    load_meshes=True,                 # 如果你想渲染机器人模型
    build_scene_graph=True           # 如果你需要进行可视化或碰撞检测
)

robot = pk.Robot.from_urdf(urdf)
robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

# motion_data, _ = load_motion_from_pkl(
#     "/home/wzh-2004/3DPOSE_TEST/human2humanoid/get_6D_pose_hand_elbow_trajectory_test.pkl",
#     motion_key="0-ACCAD_Female1General_c3d_A6- lift box t2_poses"
# )
motion_data, _ = load_motion_from_pkl(
    "/home/wzh-2004/3DPOSE_TEST/human2humanoid/data/new_robot/amass_all.pkl",
    motion_key="0-ACCAD_Female1General_c3d_A6- lift box t2_poses"
)

# 提取轨迹
# traj_local = extract_tcp_elbow_from_motion(motion_data, to_shoulder=True)
traj_root = extract_tcp_elbow_from_amass_motion(motion_data, to_shoulder=False)

target_link_name = "L_hand_base_link"
target_elbow_link_name = "left_elbow_pitch_link"
# Define the trajectory problem:
# - number of timesteps, timestep size
timesteps, dt = 25, 0.02
# - the start and end poses.
start_pos, end_pos = np.array([0.5, -0.3, 0.2]), np.array([0.5, 0.3, 0.2])

# Define the obstacles:
# - Ground
ground_coll = pk.collision.HalfSpace.from_point_and_normal(
    np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
)
# - Wall
wall_height = 0.4
wall_width = 0.1
wall_length = 0.4
wall_intervals = np.arange(start=0.3, stop=wall_length + 0.3, step=0.05)
translation = np.concatenate(
    [
        wall_intervals.reshape(-1, 1),
        np.full((wall_intervals.shape[0], 1), 0.0),
        np.full((wall_intervals.shape[0], 1), wall_height / 2),
    ],
    axis=1,
)
wall_coll = pk.collision.Capsule.from_radius_height(
    position=translation,
    radius=np.full((translation.shape[0], 1), wall_width / 2),
    height=np.full((translation.shape[0], 1), wall_height),
)
world_coll = [ground_coll, wall_coll]

tcp_traj = traj_root['left_tcp']
elbow_traj = traj_root['left_elbow']
num_frames = len(elbow_traj)

# 取 num_frames 的中位数
median_frame = num_frames // 2

start_raw_pos = traj_root['left_tcp'][0]['pos']
start_raw_rot = traj_root['left_tcp'][0]['rot']
end_raw_pos = traj_root['left_tcp'][median_frame]['pos']
end_raw_rot = traj_root['left_tcp'][median_frame]['rot']
start_down_wxyz, start_pos = start_raw_rot, start_raw_pos
end_down_wxyz, end_pos = end_raw_rot, end_raw_pos

sample_indices = np.linspace(0, median_frame - 1, num=25, dtype=int)
target_elbow_position = np.array([elbow_traj[i]["pos"] for i in sample_indices])
target_elbow_rot_quat = np.array([elbow_traj[i]["rot"] for i in sample_indices])

# 修正后的
# start_down_wxyz, start_pos = correct_to_world(start_raw_rot, start_raw_pos)
# end_down_wxyz, end_pos = correct_to_world(end_raw_rot, end_raw_pos)
print(f"Corrected Start Position: {start_pos}, End Position: {end_pos}")
print(f"Corrected Start Rot (wxyz): {start_down_wxyz}, End Rot (wxyz): {end_down_wxyz}")

# target_elbow_position = []
# target_elbow_rot_quat = []

# for i in sample_indices:
#     raw_pos = elbow_traj[i]["pos"]
#     raw_rot = elbow_traj[i]["rot"]
#     rot_wxyz, pos_corrected = correct_to_world(raw_rot, raw_pos)
#     target_elbow_position.append(pos_corrected)
#     target_elbow_rot_quat.append(rot_wxyz)

# target_elbow_position = np.array(target_elbow_position)      # shape: (25, 3)
# target_elbow_rot_quat = np.array(target_elbow_rot_quat)      # shape: (25, 4)

traj = pks.solve_trajopt(
    robot,
    robot_coll,
    world_coll,
    target_link_name,
    start_pos,
    start_down_wxyz,
    end_pos,
    end_down_wxyz,
    timesteps,
    dt,
    target_elbow_position,
    target_elbow_rot_quat,
    target_elbow_link_name,
)
traj = np.array(traj)

# Visualize!
server = viser.ViserServer()
urdf_vis = ViserUrdf(server, urdf)
server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)
server.scene.add_mesh_trimesh(
    "wall_box",
    trimesh.creation.box(
        extents=(wall_length, wall_width, wall_height),
        transform=trimesh.transformations.translation_matrix(
            np.array([0.5, 0.0, wall_height / 2])
        ),
    ),
)

for name, pos, rot in zip(
    ["start", "end", "elbow"],
    [start_pos, end_pos, target_elbow_position[-1]],
    [start_down_wxyz, end_down_wxyz, target_elbow_rot_quat[-1]],
):
    server.scene.add_frame(
        f"/{name}",
        position=pos,
        wxyz=rot,
        axes_length=0.05,
        axes_radius=0.01,
    )

slider = server.gui.add_slider(
    "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
)
playing = server.gui.add_checkbox("Playing", initial_value=True)

while True:
    if playing.value:
        slider.value = (slider.value + 1) % timesteps

    urdf_vis.update_cfg(traj[slider.value])
    time.sleep(1.0 / 10.0)