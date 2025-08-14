import time
import os
import joblib
import torch
import viser
import numpy as np

from yourdfpy import URDF
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

import pyroki_snippets as pks
import pyroki as pk
# 确保导入的是您正在使用的MLP版NNmodel
from NNmodel_improved import ElbowNNWrapper 

# --- 辅助函数 ---
def load_motion_from_pkl(all_data: dict, motion_key: str):
    if motion_key not in all_data: raise ValueError(f"❌ motion_key='{motion_key}' 不在数据中。")
    print(f"🎯 已选择 motion: {motion_key}")
    return all_data[motion_key]

def extract_all_trajectories(motion_data: dict):
    frames = motion_data["frame"]
    return {
        "left_hand_pos": frames["left_hand_root_pos"], "left_hand_rot": frames["left_hand_root_rot"],
        "right_hand_pos": frames["right_hand_root_pos"], "right_hand_rot": frames["right_hand_root_rot"],
        "left_elbow_pos": frames["left_elbow_root_pos"], "left_elbow_rot": frames["left_elbow_root_rot"],
        "right_elbow_pos": frames["right_elbow_root_pos"], "right_elbow_rot": frames["right_elbow_root_rot"],
        "left_hand_shoulder_pos": frames["left_hand_shoulder_pos"], "left_hand_shoulder_rot": frames["left_hand_shoulder_rot"],
        "right_hand_shoulder_pos": frames["right_hand_shoulder_pos"], "right_hand_shoulder_rot": frames["right_hand_shoulder_rot"],
        "left_elbow_shoulder_pos": frames["left_elbow_shoulder_pos"], "left_elbow_shoulder_rot": frames["left_elbow_shoulder_rot"],
        "right_elbow_shoulder_pos": frames["right_elbow_shoulder_pos"], "right_elbow_shoulder_rot": frames["right_elbow_shoulder_rot"],
    }

def root_to_shoulder_pose(pose_root, shoulder_pose_root):
    pos_root, quat_xyzw_root = pose_root
    shoulder_pos, shoulder_quat_xyzw = shoulder_pose_root
    if np.isnan(pos_root).any() or np.isnan(quat_xyzw_root).any() or np.isnan(shoulder_pos).any() or np.isnan(shoulder_quat_xyzw).any():
        return np.zeros(3), np.array([0., 0., 0., 1.])
    R_w, R_s = R.from_quat(quat_xyzw_root), R.from_quat(shoulder_quat_xyzw)
    pos_local = R_s.inv().apply(pos_root - shoulder_pos)
    return pos_local, (R_s.inv() * R_w).as_quat()

def shoulder_to_root(pose_local, shoulder_pose_world):
    pos_local, quat_xyzw_local = pose_local
    shoulder_pos, shoulder_quat_xyzw = shoulder_pose_world
    if np.isnan(pos_local).any() or np.isnan(quat_xyzw_local).any(): return np.zeros(3), np.array([1., 0., 0., 0.])
    R_s, R_l = R.from_quat(shoulder_quat_xyzw), R.from_quat(quat_xyzw_local)
    pos_w = R_s.apply(pos_local) + shoulder_pos
    return pos_w, np.roll((R_s * R_l).as_quat(), 1) # 返回 wxyz

def get_link_pose(robot, q, link_name):
    idx = robot.links.names.index(link_name)
    pose = robot.forward_kinematics(q[None])[0, idx]
    return np.array(pose[4:]), np.roll(np.array(pose[:4]), -1)

def main():
    # --- [核心修正] 路径已更新为您指定的文件 ---
    model_path = "/home/hjj/human2humanoid/elbow_mlp_model_10.pth"
    pkl_path = "/home/hjj/human2humanoid/data/new_robot/amass_test_10_test_10.pkl"
    urdf_path = "/home/hjj/human2humanoid/resources/robots/h1_2/urdf/h1_2.urdf"
    mesh_dir = os.path.dirname(urdf_path)
    # ------------------------------------------
    
    urdf = URDF.load(urdf_path, mesh_dir=mesh_dir)
    robot = pk.Robot.from_urdf(urdf)
    default_cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=4, height=4)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    
    pred_elbow_vis = server.scene.add_icosphere("/pred_elbow", radius=0.03, color=(255, 0, 0))
    gt_elbow_vis = server.scene.add_icosphere("/gt_elbow", radius=0.03, color=(0, 255, 0))
    
    all_motion_data = joblib.load(pkl_path)
    all_motion_keys = list(all_motion_data.keys())
    gui_motion_dropdown = server.gui.add_dropdown("选择动作", options=all_motion_keys, initial_value=all_motion_keys[0])
    is_playing_checkbox = server.gui.add_checkbox("暂停 (Pause)", initial_value=False)

    model = ElbowNNWrapper()
    model.load(model_path)
    
    t = 0
    current_motion_key = ""
    traj_data = {}
    solution_previous = default_cfg.copy()
    
    while True:
        if gui_motion_dropdown.value != current_motion_key:
            current_motion_key = gui_motion_dropdown.value
            motion_data = load_motion_from_pkl(all_motion_data, current_motion_key)
            traj_data = extract_all_trajectories(motion_data)
            num_frames = len(traj_data["left_hand_pos"])
            t = 0
            solution_previous = default_cfg.copy()

        if t >= num_frames: t = 0
        if is_playing_checkbox.value: time.sleep(0.03); continue
        
        shoulder_pos, shoulder_quat_xyzw = get_link_pose(robot, solution_previous, "left_shoulder_pitch_link")

        # --- 为MLP模型准备输入 (修复了bug的稳定版本) ---
        seq_feats = []
        for k in range(4, -1, -1):
            idx = max(0, t - k)
            feat_list = []
            for part in ["left_hand", "right_hand", "left_elbow", "right_elbow"]:
                pos_root, rot_root_wxyz = traj_data[f"{part}_pos"][idx], traj_data[f"{part}_rot"][idx]
                pos_s, rot_s_xyzw = root_to_shoulder_pose((pos_root, np.roll(rot_root_wxyz, -1)), (shoulder_pos, shoulder_quat_xyzw))
                feat_list.extend([pos_s, rot_s_xyzw])
            seq_feats.append(np.concatenate(feat_list))
        input_seq_shoulder = np.stack(seq_feats, axis=0)

        next_idx = min(t + 1, num_frames - 1)
        next_tcp_list = []
        for part in ["left_hand", "right_hand"]:
            pos_root, rot_root_wxyz = traj_data[f"{part}_pos"][next_idx], traj_data[f"{part}_rot"][next_idx]
            pos_s, rot_s_xyzw = root_to_shoulder_pose((pos_root, np.roll(rot_root_wxyz, -1)), (shoulder_pos, shoulder_quat_xyzw))
            next_tcp_list.extend([pos_s, rot_s_xyzw])
        next_tcp_shoulder = np.concatenate(next_tcp_list)
        
        input_tensor = torch.cat([
            torch.from_numpy(input_seq_shoulder.flatten()),
            torch.from_numpy(next_tcp_shoulder)
        ]).float()

        # --- 模型预测 ---
        _predictions = model.predict(input_tensor)
        if torch.isnan(_predictions).any(): t+=1; continue

        _pred_pos_s = _predictions[:3].cpu().numpy()
        _pred_rot_xyzw_s = _predictions[3:7].cpu().numpy()
        
        _pred_pos_root, _pred_rot_root_wxyz = shoulder_to_root(
            (_pred_pos_s, _pred_rot_xyzw_s), (shoulder_pos, shoulder_quat_xyzw)
        )
        
        # --- 更新可视化和IK目标 ---
        gt_elbow_vis.position = traj_data["left_elbow_pos"][t]
        pred_elbow_vis.position = _pred_pos_root
        
        target_tcp_pos = traj_data["left_hand_pos"][t]
        target_tcp_wxyz = traj_data["left_hand_rot"][t]

        solution = pks.solve_ik(
            robot=robot, target_link_name="L_hand_base_link",
            target_position=target_tcp_pos, target_wxyz=target_tcp_wxyz,
            target_elbow_position=_pred_pos_root,
            target_elbow_rot_quat=_pred_rot_root_wxyz,
            target_elbow_link_name="left_elbow_pitch_link", init_q=solution_previous,
        )
        
        if np.isnan(solution).any(): solution = default_cfg.copy()
        solution_previous = solution
        
        server.scene.add_frame("/animation_target", position=target_tcp_pos, wxyz=target_tcp_wxyz, axes_length=0.1)
        urdf_vis.update_cfg(solution)
        time.sleep(1.0 / 30.0)
        t += 1

if __name__ == "__main__":
    main()