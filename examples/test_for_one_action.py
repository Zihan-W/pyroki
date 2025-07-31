import pyroki as pk
import joblib
import os
from yourdfpy import URDF
import time

import numpy as np
import pyroki as pk
import viser
import jaxlie
from pyroki.collision import HalfSpace, RobotCollision, Sphere
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

import pyroki_snippets as pks
urdf_path = "/home/wzh-2004/3DPOSE_TEST/human2humanoid/resources/robots/h1_2/urdf/h1_2.urdf"
mesh_dir = "/home/wzh-2004/3DPOSE_TEST/human2humanoid/resources/robots/h1_2/meshes"
urdf = URDF.load(
    urdf_path,
    mesh_dir=mesh_dir,                 # 用于解析 mesh 文件路径
    load_meshes=True,                 # 如果你想渲染机器人模型
    build_scene_graph=True           # 如果你需要进行可视化或碰撞检测
)
robot = pk.Robot.from_urdf(urdf)

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

def extract_tcp_elbow_from_motion(motion_data: dict, use_local=True):
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
        if use_local:
            left_tcp = {
                "pos": frame["left_shoulder_local"]["tcp_pos"],
                "rot": frame["left_shoulder_local"]["tcp_rot"]
            }
            right_tcp = {
                "pos": frame["right_shoulder_local"]["tcp_pos"],
                "rot": frame["right_shoulder_local"]["tcp_rot"]
            }
            left_elbow = {
                "pos": frame["left_shoulder_local"]["elbow_pos"],
                "rot": frame["left_shoulder_local"]["elbow_rot"]
            }
            right_elbow = {
                "pos": frame["right_shoulder_local"]["elbow_pos"],
                "rot": frame["right_shoulder_local"]["elbow_rot"]
            }
        else:
            left_tcp = frame["left_tcp_world"]
            right_tcp = frame["right_tcp_world"]
            left_elbow = {
                "pos": frame["left_elbow_world"],
                "rot": None  # 世界坐标下没有提供 elbow 旋转
            }
            right_elbow = {
                "pos": frame["right_elbow_world"],
                "rot": None
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


# _, motion_keys = load_motion_from_pkl("/home/wzh-2004/3DPOSE_TEST/human2humanoid/hand_elbow_trajectory_test.pkl")
# motion_data, _ = load_motion_from_pkl("hand_elbow_trajectory_test.pkl", motion_key="walk_01")

motion_data, _ = load_motion_from_pkl(
    "/home/wzh-2004/3DPOSE_TEST/human2humanoid/hand_elbow_trajectory_test.pkl",
    motion_key="0-ACCAD_Female1General_c3d_A6- lift box t2_poses"
)

# 提取轨迹
traj_local = extract_tcp_elbow_from_motion(motion_data, use_local=True)
traj_global = extract_tcp_elbow_from_motion(motion_data, use_local=False)

target_link_name = "left_wrist_pitch_link"
target_elbow_link_name = "left_elbow_pitch_link"
robot_coll = RobotCollision.from_urdf(urdf)
plane_coll = HalfSpace.from_point_and_normal(
    np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
)
sphere_coll = Sphere.from_center_and_radius(
    np.array([0.0, 0.0, 0.0]), np.array([0.05])
)

# Set up visualizer.
server = viser.ViserServer()
server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

# Create interactive controller for IK target.
ik_target_handle = server.scene.add_transform_controls(
    "/ik_target", scale=0.2, position=(0.5, 0.0, 0.5), wxyz=(0, 0, 1, 0)
)

# Create interactive controller and mesh for the sphere obstacle.
sphere_handle = server.scene.add_transform_controls(
    "/obstacle", scale=0.2, position=(0.4, 0.3, 0.4)
)
server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())

timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

while True:
    for step in range(len(traj_global['left_tcp'])):
        start_time = time.time()
        target_elbow_pose = traj_global['left_elbow'][step]['pos']

        sphere_coll_world_current = sphere_coll.transform_from_wxyz_position(
            wxyz=np.array(sphere_handle.wxyz),
            position=np.array(sphere_handle.position),
        )

        world_coll_list = [plane_coll, sphere_coll_world_current]
        solution = pks.solve_ik_with_collision(
            robot=robot,
            coll=robot_coll,
            world_coll_list=world_coll_list,
            target_link_name=target_link_name,
            target_elbow_link_name = target_elbow_link_name,
            target_position=np.array(traj_global['left_tcp'][step]['pos']),
            target_wxyz=np.array(traj_global['left_tcp'][step]['rot']),
            target_elbow_position = target_elbow_pose,
        )

        # Update timing handle.
        timing_handle.value = (time.time() - start_time) * 1000

        # Update visualizer.
        urdf_vis.update_cfg(solution)
        time.sleep(0.5)
    # start_time = time.time()

    # sphere_coll_world_current = sphere_coll.transform_from_wxyz_position(
    #     wxyz=np.array(sphere_handle.wxyz),
    #     position=np.array(sphere_handle.position),
    # )

    # world_coll_list = [plane_coll, sphere_coll_world_current]
    # solution = pks.solve_ik_with_collision(
    #     robot=robot,
    #     coll=robot_coll,
    #     world_coll_list=world_coll_list,
    #     target_link_name=target_link_name,
    #     target_position=np.array(ik_target_handle.position),
    #     target_wxyz=np.array(ik_target_handle.wxyz),
    #     target_elbow_pose = target_elbow_pose,
    # )

    # # Update timing handle.
    # timing_handle.value = (time.time() - start_time) * 1000

    # # Update visualizer.
    # urdf_vis.update_cfg(solution)