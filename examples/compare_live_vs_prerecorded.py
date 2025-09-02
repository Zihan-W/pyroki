# /home/hjj/pyroki/examples/compare_live_vs_prerecorded_urdf_hand_humanShoulderStatic.py
import os
import time
import joblib
import torch
import numpy as np
from yourdfpy import URDF
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import viser
from viser.extras import ViserUrdf

import pyroki as pk
import pyroki_snippets as pks
from NNmodel_comprehensive_exp import ElbowNNWrapper

# ======================== 配置 ========================
MODEL_INFO = {
    "name": "STAttention_Optimal",
    "model_config": {"type": "st_attention","params": {"hidden_dim": 1792, "num_layers": 2, "nhead": 8, "dropout_rate": 0.2}},
    "train_params": {"history_len": 5},
    "path": "/home/hjj/human2humanoid/models_final_comparison_with_film/exp_STAttention_for_eval.pth",
    "stats_path": "/home/hjj/human2humanoid/models_final_comparison_with_film/exp_STAttention_for_eval_stats.pkl",
}

CONFIG = {
    "pkl_path": "/home/hjj/human2humanoid/data/new_robot/amass_all_corrected_new_full_final.pkl",
    "urdf_path": "/home/hjj/human2humanoid/resources/robots/h1_2/urdf/h1_2.urdf",

    "sample_motion_key":"0-SFU_0018_0018_Walking001_poses",
    "sample_frame_index": 100,

    "ik_weights": {
        "baseline":  {"pos_weight": 80.0, "ori_weight": 30.0, "elbow_pos_weight": 0.0,  "elbow_ori_weight": 0.0,  "smoothness_weight": 12.0},
        "optimized": {"pos_weight": 80.0, "ori_weight": 30.0, "elbow_pos_weight": 120.0, "elbow_ori_weight": 5.0, "smoothness_weight": 12.0},
    },

    "visual_align_human_shoulder_z": True,
    "USE_NEXT_FRAME_TCP_FOR_IK": True,

    "auto_choose_reference": True,
    "min_neck_pelvis_dist": 0.12,
    "min_shoulder_width":  0.18,

    "scale_human_for_eval": True,
    "scale_human_for_ik_target": False,
    "align_human_to_robot_shoulder": True,

    # 方案B：工具变换（把“想要的TCP目标”换算成“底座链接(L_hand_base_link)目标”）
    "enable_tool_transform_in_ik": True,       # 打开后，IK 目标用底座=TCP*Tool^{-1} 的换算
    "enable_tool_transform_in_eval": False,    # 如你的评估目标是底座点而非TCP点，可打开此项做同样换算
    "tool_axes": "z",                          # 工具旋转轴，e.g. "z"、"xyz"
    "tool_angle_deg": 90.0,                    # 工具旋转角度（若多轴可用列表/元组）
    "tool_xyz": [0.0, 0.0, 0.0],               # 工具相对底座的平移（以底座坐标系定义），单位m

    # 可视化：同时画底座轴与TCP轴
    "show_axes_in_vis": True,
    "tcp_axes_offset_world": [0.06, 0.0, 0.0], # TCP坐标轴在世界中偏移，避免重叠，便于观察
}

ROBOT_OFFSET = np.array([0.0, 0.0, 0.0])

# ======================== 标准化（肩系） ========================
def _normalize_fallback_old(points_in_world: dict):
    pelvis = points_in_world["pelvis"]
    pts_T = {k: v - pelvis for k, v in points_in_world.items()}
    up = pts_T["neck"] - pts_T["pelvis"]; up = up / (np.linalg.norm(up) + 1e-8)
    right_raw = pts_T["right_shoulder"] - pts_T["left_shoulder"]
    right = right_raw - np.dot(right_raw, up) * up; right = right / (np.linalg.norm(right) + 1e-8)
    front = np.cross(right, up)
    R_undo = np.stack([right, up, front], axis=1).T
    names = list(pts_T.keys()); vals = np.stack([pts_T[n] for n in names], axis=0)
    vals_aligned = (R_undo @ vals.T).T
    R_target = R.from_euler("yx", [-90.0, -90.0], degrees=True)
    vals_target = R_target.apply(vals_aligned)
    R_flip = R.from_euler("z", 180.0, degrees=True)
    vals_final = R_flip.apply(vals_target)
    return {n: v for n, v in zip(names, vals_final)}

def normalize_human_points_to_robot_convention(points_in_world: dict):
    pelvis = points_in_world["pelvis"]
    pts_T = {k: v - pelvis for k, v in points_in_world.items()}
    LS = pts_T["left_shoulder"]; RS = pts_T["right_shoulder"]; NK = pts_T["neck"]
    right = RS - LS; right_n = np.linalg.norm(right)
    if right_n < 1e-8: return _normalize_fallback_old(points_in_world)
    right = right / right_n
    shoulder_mid = 0.5 * (LS + RS)
    up_raw = NK - shoulder_mid; up_raw = up_raw - np.dot(up_raw, right) * right
    up_n = np.linalg.norm(up_raw)
    if up_n < 1e-8: return _normalize_fallback_old(points_in_world)
    up = up_raw / up_n
    front = np.cross(right, up); front_n = np.linalg.norm(front)
    if front_n < 1e-8: return _normalize_fallback_old(points_in_world)
    front = front / front_n
    R_undo = np.stack([right, up, front], axis=1).T
    names = list(pts_T.keys()); vals = np.stack([pts_T[n] for n in names], axis=0)
    vals_aligned = (R_undo @ vals.T).T
    R_target = R.from_euler("yx", [-90.0, -90.0], degrees=True)
    vals_target = R_target.apply(vals_aligned)
    R_flip = R.from_euler("z", 180.0, degrees=True)
    vals_final = R_flip.apply(vals_target)
    return {n: v for n, v in zip(names, vals_final)}

# ======================== 参考帧选择（稳健） ========================
def torso_plausibility_score(hw: dict) -> float:
    d_np = np.linalg.norm(hw["neck"] - hw["pelvis"])
    d_sw = np.linalg.norm(hw["right_shoulder"] - hw["left_shoulder"])
    if not np.isfinite(d_np + d_sw): return -1.0
    return float(d_np + d_sw)

def choose_reference_index(raw_hw_list: list, min_np: float, min_sw: float, fallback_idx: int) -> int:
    cand = []
    for i, hw in enumerate(raw_hw_list):
        d_np = np.linalg.norm(hw["neck"] - hw["pelvis"])
        d_sw = np.linalg.norm(hw["right_shoulder"] - hw["left_shoulder"])
        if d_np >= min_np and d_sw >= min_sw:
            cand.append((torso_plausibility_score(hw), i))
    if cand:
        cand.sort(reverse=True); return cand[0][1]
    all_scores = [(torso_plausibility_score(hw), i) for i, hw in enumerate(raw_hw_list)]
    all_scores.sort(reverse=True)
    if all_scores and np.isfinite(all_scores[0][0]) and all_scores[0][0] > 0: return all_scores[0][1]
    return fallback_idx

# ======================== 机器人/姿态工具 ========================
def get_link_pose(robot: pk.Robot, q: np.ndarray, link_name: str):
    idx = robot.links.names.index(link_name)
    pose = robot.forward_kinematics(q[None])[0, idx]
    pos = np.array(pose[4:], dtype=float); quat_wxyz = np.array(pose[:4], dtype=float)
    quat_xyzw = np.roll(quat_wxyz, -1); return pos, quat_xyzw

def shoulder_to_root(pose_local, shoulder_pose_world):
    pos_local, quat_xyzw_local = pose_local; shoulder_pos, shoulder_quat_xyzw = shoulder_pose_world
    R_s = R.from_quat(shoulder_quat_xyzw); R_l = R.from_quat(quat_xyzw_local)
    pos_w = R_s.apply(pos_local) + shoulder_pos; quat_wxyz = np.roll((R_s * R_l).as_quat(), 1); return pos_w, quat_wxyz

def make_q_dict_for_urdf(urdf: URDF, robot: pk.Robot, q_vec: np.ndarray):
    name_to_idx = {name: i for i, name in enumerate(robot.joints.names)}
    q_dict = {}
    for j in urdf.actuated_joints:
        if j.name in name_to_idx:
            q_dict[j.name] = float(q_vec[name_to_idx[j.name]])
    return q_dict

def prev_hand_wxyz(robot: pk.Robot, q_prev: np.ndarray):
    _, quat_b_xyzw_prev = get_link_pose(robot, q_prev, "L_hand_base_link")
    return np.roll(R.from_quat(quat_b_xyzw_prev).as_quat(), 1)

def shift_forward(arr):
    return np.concatenate([arr[1:], arr[-1:]], axis=0)

# ======================== 机器人名义上肢段长 ========================
def robot_nominal_upper_limb_lengths(robot):
    q_mid = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    p_sh = get_link_pose(robot, q_mid, "left_shoulder_pitch_link")[0]
    p_el = get_link_pose(robot, q_mid, "left_elbow_pitch_link")[0]
    p_wr = get_link_pose(robot, q_mid, "left_wrist_pitch_link")[0]
    p_hd = get_link_pose(robot, q_mid, "L_hand_base_link")[0]
    return {"Lr_ua": float(np.linalg.norm(p_el - p_sh)), "Lr_fa": float(np.linalg.norm(p_wr - p_el)), "Lr_ha": float(np.linalg.norm(p_hd - p_wr))}

# ======================== 逐帧提取左臂局部段向量 ========================
def extract_per_frame_local_upper_limb(human_static_seq):
    T = len(human_static_seq)
    v_ua_list, v_fa_list, v_ha_list = [], [], []
    for i in range(T):
        LS = human_static_seq[i]["left_shoulder"]
        LE = human_static_seq[i]["left_elbow"]
        LW = human_static_seq[i]["left_wrist"]
        LH = human_static_seq[i]["left_hand"]
        v_ua_list.append(LE - LS)
        v_fa_list.append(LW - LE)
        v_ha_list.append(LH - LW)
    return np.stack(v_ua_list), np.stack(v_fa_list), np.stack(v_ha_list)

# ======================== 构造“比例化/非比例化”的世界静态（逐帧左臂） ========================
def build_world_aligned_human_static(
    human_static_seq, t_ref, robot_shoulder_pos_world, robot_lengths,
    scale_to_robot=True, enforce_axis_align=True,
):
    T = len(human_static_seq)
    ref = human_static_seq[t_ref]
    shoulder_width = float(np.linalg.norm(ref["right_shoulder"] - ref["left_shoulder"]))
    pelvis_dz = float(ref["pelvis"][2] - ref["left_shoulder"][2])

    v_ua_list, v_fa_list, v_ha_list = extract_per_frame_local_upper_limb(human_static_seq)

    def unit_rows(V):
        n = np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
        return V / n

    if scale_to_robot:
        u_ua = unit_rows(v_ua_list)
        u_fa = unit_rows(v_fa_list)
        u_ha = unit_rows(v_ha_list)

    targets = []
    for i in range(T):
        LS_w = robot_shoulder_pos_world + ROBOT_OFFSET
        if enforce_axis_align:
            RS_w = LS_w + np.array([0.0, -shoulder_width, 0.0])
            PV_w = np.array([0.0, 0.0, LS_w[2] + pelvis_dz])
        else:
            RS_w = LS_w + (ref["right_shoulder"] - ref["left_shoulder"])
            PV_w = LS_w + (ref["pelvis"] - ref["left_shoulder"])

        if scale_to_robot:
            LE_w = LS_w + robot_lengths["Lr_ua"] * u_ua[i]
            LW_w = LE_w + robot_lengths["Lr_fa"] * u_fa[i]
            LH_w = LW_w + robot_lengths["Lr_ha"] * u_ha[i]
        else:
            LE_w = LS_w + v_ua_list[i]
            LW_w = LE_w + v_fa_list[i]
            LH_w = LW_w + v_ha_list[i]

        targets.append({
            "pelvis": PV_w,
            "left_shoulder": LS_w,
            "right_shoulder": RS_w,
            "left_elbow": LE_w,
            "left_wrist": LW_w,
            "left_hand": LH_w,  # 注意：此处我们把“左手点”视为 TCP 目标点
        })
    return targets

# ======================== 工具变换（方案B核心） ========================
def build_tool_transform():
    # 工具变换：从“底座链接（L_hand_base_link）”到“TCP”的固定变换 T_tool
    # 返回 (R_tool, t_tool)
    axes = CONFIG.get("tool_axes", "z")
    ang = CONFIG.get("tool_angle_deg", 90.0)
    R_tool = R.from_euler(axes, ang, degrees=True)
    t_tool = np.array(CONFIG.get("tool_xyz", [0.0, 0.0, 0.0]), dtype=float)
    return R_tool, t_tool

def tcp_to_base_target(R_tcp_desired: R, p_tcp_desired: np.ndarray, R_tool: R, t_tool: np.ndarray):
    # 给定 期望的 TCP 世界姿态/位置 (R_tcp_desired, p_tcp_desired)
    # 以及 工具变换 T_tool: base->tcp (R_tool, t_tool)
    # 计算 期望的“底座链接”(L_hand_base_link)目标：
    # R_base = R_tcp * R_tool^{-1}
    # p_base = p_tcp - R_base * t_tool
    R_base = R_tcp_desired * R_tool.inv()
    p_base = p_tcp_desired - R_base.apply(t_tool)
    return R_base, p_base

# ============== 运行一条 IK 序列 =================
@torch.no_grad()
def run_ik_sequence(
    robot,
    frame,
    model,
    history_len,
    weights,
    use_next_frame_tcp_for_ik,
    world_targets_tcp,           # 这里的目标“left_hand”被认为是 TCP 点
    enable_tool_transform_in_ik, # 是否在 IK 中将 TCP 目标换算为底座目标
    R_tool: R,
    t_tool: np.ndarray,
):
    T = frame["left_hand_root_pos"].shape[0]
    q_prev = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    dof = robot.joints.num_actuated_joints
    q_seq = np.zeros((T, dof), dtype=float)
    ee_pos_seq = np.zeros((T, 3), dtype=float)
    chain_pos_seq = {k: np.zeros((T, 3), dtype=float) for k in ["shoulder", "elbow", "wrist", "hand"]}

    for i in tqdm(range(T), desc=f"IK[{weights.get('tag','')}]"):
        # 历史输入
        seq = []
        for k in range(history_len - 1, -1, -1):
            idx = max(0, i - k)
            feat = np.concatenate([
                frame["left_hand_shoulder_pos"][idx],
                frame["left_hand_shoulder_rot"][idx],
                frame["right_hand_shoulder_pos"][idx],
                frame["right_hand_shoulder_rot"][idx],
                frame["left_elbow_shoulder_pos"][idx],
                frame["left_elbow_shoulder_rot"][idx],
                frame["right_elbow_shoulder_pos"][idx],
                frame["right_elbow_shoulder_rot"][idx],
            ])
            seq.append(feat)
        x_seq = torch.from_numpy(np.stack(seq)).float()

        next_idx = min(i + 1, T - 1)
        x_next_tcp = torch.from_numpy(np.concatenate([
            frame["left_hand_shoulder_pos"][next_idx],
            frame["left_hand_shoulder_rot"][next_idx],
            frame["right_hand_shoulder_pos"][next_idx],
            frame["right_hand_shoulder_rot"][next_idx],
        ])).float()

        preds = model.predict(x_seq, x_next_tcp)
        pred_pos_local = preds[:3].cpu().numpy()
        pred_rot_local = preds[3:7].cpu().numpy()

        shoulder_pos, shoulder_quat_xyzw = get_link_pose(robot, q_prev, "left_shoulder_pitch_link")
        pred_elbow_world_pos, pred_elbow_world_quat_wxyz = shoulder_to_root(
            (pred_pos_local, pred_rot_local), (shoulder_pos, shoulder_quat_xyzw)
        )

        sel_idx = next_idx if use_next_frame_tcp_for_ik else i

        # 期望的 TCP 世界位置（来自逐帧人类目标）
        p_tcp_desired = world_targets_tcp[sel_idx]["left_hand"]

        # 朝向连续性：沿用上一帧“TCP朝向”（用上一帧底座姿态推 TCP）
        _, quat_b_xyzw_prev = get_link_pose(robot, q_prev, "L_hand_base_link")
        R_base_prev = R.from_quat(quat_b_xyzw_prev)
        R_tcp_desired = R_base_prev * R_tool  # 上一帧底座 * 工具 -> 上一帧TCP朝向

        if enable_tool_transform_in_ik:
            # 把“期望的 TCP 目标”换算成“底座目标”
            R_base_target, p_base_target = tcp_to_base_target(R_tcp_desired, p_tcp_desired, R_tool, t_tool)
        else:
            # 不使用工具换算：直接把“TCP的位置”当成“底座位置目标”，朝向用上一帧底座朝向
            R_base_target = R_base_prev
            p_base_target = p_tcp_desired

        target_wxyz_b = np.roll(R_base_target.as_quat(), 1)

        if weights.get("elbow_pos_weight", 0.0) == 0.0 and weights.get("elbow_ori_weight", 0.0) == 0.0:
            q_sol = pks.solve_ik(
                robot=robot, target_link_name="L_hand_base_link",
                target_position=p_base_target, target_wxyz=target_wxyz_b,
                init_q=q_prev, pos_weight=weights["pos_weight"], ori_weight=weights["ori_weight"],
                elbow_pos_weight=0.0, elbow_ori_weight=0.0, smoothness_weight=weights["smoothness_weight"],
            )
        else:
            q_sol = pks.solve_ik(
                robot=robot, target_link_name="L_hand_base_link",
                target_position=p_base_target, target_wxyz=target_wxyz_b,
                init_q=q_prev,
                target_elbow_position=pred_elbow_world_pos, target_elbow_rot_quat=pred_elbow_world_quat_wxyz,
                target_elbow_link_name="left_elbow_pitch_link",
                pos_weight=weights["pos_weight"], ori_weight=weights["ori_weight"],
                elbow_pos_weight=weights["elbow_pos_weight"], elbow_ori_weight=weights["elbow_ori_weight"],
                smoothness_weight=weights["smoothness_weight"],
            )

        if np.isnan(q_sol).any(): q_sol = q_prev.copy()
        q_prev = q_sol; q_seq[i] = q_sol

        # 记录链条关键点（底座链路）
        chain_pos_seq["shoulder"][i] = get_link_pose(robot, q_sol, "left_shoulder_pitch_link")[0]
        chain_pos_seq["elbow"][i]    = get_link_pose(robot, q_sol, "left_elbow_pitch_link")[0]
        chain_pos_seq["wrist"][i]    = get_link_pose(robot, q_sol, "left_wrist_pitch_link")[0]
        chain_pos_seq["hand"][i]     = get_link_pose(robot, q_sol, "L_hand_base_link")[0]
        ee_pos_seq[i] = chain_pos_seq["hand"][i]

    return {"q_seq": q_seq, "ee_pos_seq": ee_pos_seq, "chain_pos_seq": chain_pos_seq}

# ======================== 主流程 ========================
def main():
    print("=" * 80)
    print("URDF 手端（L_hand_base_link）+ 躯干锁定 + 左肩对齐 + 逐帧左臂 + 可选比例化 + 工具变换(方案B)")
    print("=" * 80)

    all_motion = joblib.load(CONFIG["pkl_path"])
    motion_key = list(all_motion.keys())[0] if CONFIG["sample_motion_key"] is None else CONFIG["sample_motion_key"]
    motion = all_motion[motion_key]; frame = motion["frame"]
    T = frame["left_hand_root_pos"].shape[0]
    print(f"[INFO] motion: {motion_key} | T={T}")

    urdf_path = CONFIG["urdf_path"]
    urdf = URDF.load(urdf_path, mesh_dir=os.path.dirname(urdf_path))
    robot = pk.Robot.from_urdf(urdf)

    model = ElbowNNWrapper(model_config=MODEL_INFO["model_config"])
    model.load(MODEL_INFO["path"], MODEL_INFO["stats_path"])
    history_len = MODEL_INFO["train_params"]["history_len"]

    # 原始世界坐标序列
    raw_hw_list = []
    for i in range(T):
        hw = {
            "pelvis": motion["root_trans_offset"][i],
            "neck": frame["neck_root_pos"][i],
            "left_shoulder": frame["left_shoulder_root_pos"][i],
            "right_shoulder": frame["right_shoulder_root_pos"][i],
            "left_elbow": frame["left_elbow_root_pos"][i],
            "left_wrist": frame["left_wrist_root_pos"][i],
            "left_hand": frame["left_hand_root_pos"][i],
        }
        raw_hw_list.append(hw)

    # 参考帧
    if CONFIG.get("auto_choose_reference", True):
        t_ref = choose_reference_index(raw_hw_list, CONFIG["min_neck_pelvis_dist"], CONFIG["min_shoulder_width"], min(CONFIG["sample_frame_index"], T - 1))
        print(f"[REF] auto_choose_reference=True -> ref_index={t_ref}")
    else:
        t_ref = min(CONFIG.get("sample_frame_index", 0), T - 1)
        print(f"[REF] auto_choose_reference=False -> ref_index={t_ref}")

    # 标准化 + 躯干锁定（仅左臂三点保留动态）
    human_norm_seq = [normalize_human_points_to_robot_convention(hw) for hw in raw_hw_list]
    human_static_seq = []
    ref_pose = human_norm_seq[t_ref]
    dynamic_keys = {"left_elbow", "left_wrist", "left_hand"}
    for i in range(T):
        cur = human_norm_seq[i].copy()
        for k in cur.keys():
            if k not in dynamic_keys:
                cur[k] = ref_pose[k]
        human_static_seq.append(cur)

    # 机器人名义长度
    robot_lengths = robot_nominal_upper_limb_lengths(robot)

    # 基准机器人左肩（中位姿）
    q_mid = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    robot_shoulder_mid = get_link_pose(robot, q_mid, "left_shoulder_pitch_link")[0]

    # 世界坐标静态参考（逐帧左臂；评估可选比例化；IK 可选比例化）
    world_static_eval_tcp = build_world_aligned_human_static(
        human_static_seq, t_ref, robot_shoulder_mid, robot_lengths,
        scale_to_robot=CONFIG["scale_human_for_eval"], enforce_axis_align=CONFIG["align_human_to_robot_shoulder"],
    )
    world_static_ik_tcp = build_world_aligned_human_static(
        human_static_seq, t_ref, robot_shoulder_mid, robot_lengths,
        scale_to_robot=CONFIG["scale_human_for_ik_target"], enforce_axis_align=CONFIG["align_human_to_robot_shoulder"],
    )

    # 时间对齐（下一帧 TCP）
    if CONFIG["USE_NEXT_FRAME_TCP_FOR_IK"]:
        def shift_list(lst): return lst[1:] + lst[-1:]
        world_static_eval_tcp = shift_list(world_static_eval_tcp)
        world_static_ik_tcp   = shift_list(world_static_ik_tcp)

    # 工具变换
    R_tool, t_tool = build_tool_transform()
    print(f"[Tool] axes={CONFIG['tool_axes']} angle_deg={CONFIG['tool_angle_deg']} t={CONFIG['tool_xyz']}")
    print(f"[Tool] enable_in_ik={CONFIG['enable_tool_transform_in_ik']} enable_in_eval={CONFIG['enable_tool_transform_in_eval']}")

    # 运行 IK（把“想要的TCP目标”换算成“底座目标”）
    use_next = CONFIG["USE_NEXT_FRAME_TCP_FOR_IK"]
    seq_baseline = run_ik_sequence(
        robot, frame, model, history_len,
        {**CONFIG["ik_weights"]["baseline"], "tag": "baseline"},
        use_next, world_static_ik_tcp,
        enable_tool_transform_in_ik=CONFIG["enable_tool_transform_in_ik"],
        R_tool=R_tool, t_tool=np.array(t_tool, dtype=float),
    )
    seq_optimized = run_ik_sequence(
        robot, frame, model, history_len,
        {**CONFIG["ik_weights"]["optimized"], "tag": "optimized"},
        use_next, world_static_ik_tcp,
        enable_tool_transform_in_ik=CONFIG["enable_tool_transform_in_ik"],
        R_tool=R_tool, t_tool=np.array(t_tool, dtype=float),
    )

    # 评估（默认评估 TCP 目标 vs 机器人底座实际位置：若要评估底座目标，打开 enable_tool_transform_in_eval）
    eval_targets = world_static_eval_tcp
    if CONFIG["enable_tool_transform_in_eval"]:
        # 把 TCP 目标换成“底座目标点”（仅位置；朝向指标我们用连杆方向余弦，不依赖末端朝向）
        eval_targets = []
        for d in world_static_eval_tcp:
            # 期望TCP位置/朝向：位置来自 d["left_hand"]；朝向未用
            p_tcp = d["left_hand"]
            # 选用中位姿底座朝向推一个参考（朝向对位置换算只影响 t_tool 旋转方向；如 t_tool 非零才有意义）
            R_base_ref = R.from_euler("xyz", [0, 0, 0], degrees=True)
            p_base = p_tcp - R_base_ref.apply(t_tool)
            eval_targets.append({**d, "left_hand": p_base})

    H_shoulder = np.stack([d["left_shoulder"] for d in eval_targets], axis=0)
    H_elbow    = np.stack([d["left_elbow"] for d in eval_targets], axis=0)
    H_wrist    = np.stack([d["left_wrist"] for d in eval_targets], axis=0)
    H_hand     = np.stack([d["left_hand"] for d in eval_targets], axis=0)

    def end_effector_mse_rmse(ee_pos_seq, H_hand):
        diff = ee_pos_seq - H_hand
        mse = float(np.mean(np.sum(diff ** 2, axis=1))); rmse = float(np.sqrt(mse))
        return mse, rmse

    def chain_cosine_means(chain_pos_seq, H_shoulder, H_elbow, H_wrist, H_hand):
        R_sh = chain_pos_seq["shoulder"]; R_el = chain_pos_seq["elbow"]
        R_wr = chain_pos_seq["wrist"];    R_hd = chain_pos_seq["hand"]
        R_ua, R_fa, R_ha = R_el - R_sh, R_wr - R_el, R_hd - R_wr
        H_ua, H_fa, H_ha = H_elbow - H_shoulder, H_wrist - H_elbow, H_hand - H_wrist
        def cos_batch(A, B):
            num = np.sum(A * B, axis=1); na = np.linalg.norm(A, axis=1) + 1e-8; nb = np.linalg.norm(B, axis=1) + 1e-8
            return num / (na * nb)
        return float(np.mean(cos_batch(R_ua, H_ua))), float(np.mean(cos_batch(R_fa, H_fa))), float(np.mean(cos_batch(R_ha, H_ha)))

    mse_b, rmse_b = end_effector_mse_rmse(seq_baseline["ee_pos_seq"], H_hand)
    mse_o, rmse_o = end_effector_mse_rmse(seq_optimized["ee_pos_seq"], H_hand)
    cos_b = chain_cosine_means(seq_baseline["chain_pos_seq"], H_shoulder, H_elbow, H_wrist, H_hand)
    cos_o = chain_cosine_means(seq_optimized["chain_pos_seq"], H_shoulder, H_elbow, H_wrist, H_hand)

    print("\n========== 序列指标（末端=URDF 手端；左肩对齐；逐帧左臂；比例化评估={}；工具变换 in IK={}） =========="
          .format(CONFIG["scale_human_for_eval"], CONFIG["enable_tool_transform_in_ik"]))
    print(f"- ref_index: {t_ref}")
    print(f"- End-effector MSE (m^2): baseline={mse_b:.6f} | optimized={mse_o:.6f}")
    print(f"- End-effector RMSE (m):  baseline={rmse_b:.6f} | optimized={rmse_o:.6f}")
    print(f"- Cosine similarity (mean over frames):")
    print(f"  Upper Arm: baseline={cos_b[0]:.4f} | optimized={cos_o[0]:.4f}")
    print(f"  Forearm  : baseline={cos_b[1]:.4f} | optimized={cos_o[1]:.4f}")
    print(f"  Hand     : baseline={cos_b[2]:.4f} | optimized={cos_o[2]:.4f}")

    # ---------- 可视化 ----------
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=6.0, height=6.0)
    server.scene.add_frame("/world_axes", axes_length=0.4, axes_radius=0.01)
    server.gui.add_markdown(
        "### URDF 手端 + 躯干锁定 + 左肩对齐 + 逐帧左臂 + 可选比例化 + 工具变换(方案B)\n"
        f"- 比例化评估: {CONFIG['scale_human_for_eval']} | 比例化IK目标: {CONFIG['scale_human_for_ik_target']} | IK工具变换: {CONFIG['enable_tool_transform_in_ik']}"
    )
    play_toggle = server.gui.add_checkbox("Play", initial_value=True)
    fps_number = server.gui.add_number("FPS", min=1, max=120, step=1, initial_value=30)
    loop_toggle = server.gui.add_checkbox("Loop", initial_value=True)
    traj_select = server.gui.add_dropdown("Trajectory", options=["optimized", "baseline"], initial_value="optimized")
    frame_slider = server.gui.add_slider("Frame", min=0, max=T - 1, step=1, initial_value=min(CONFIG["sample_frame_index"], T - 1))
    slider_silent = False

    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.position = (float(ROBOT_OFFSET[0]), float(ROBOT_OFFSET[1]), float(ROBOT_OFFSET[2]))

    human_nodes = {
        "pelvis": "/human/pelvis",
        "lshoulder": "/human/lshoulder",
        "rshoulder": "/human/rshoulder",
        "lelbow": "/human/lelbow",
        "lwrist": "/human/lwrist",
        "lhand": "/human/lhand",
    }
    human_bones = [
        ("lshoulder", "rshoulder"),
        ("lshoulder", "lelbow"),
        ("lelbow", "lwrist"),
        ("lwrist", "lhand"),
        ("pelvis", "lshoulder"),
    ]
    human_color = (80, 160, 255)
    for path in human_nodes.values():
        server.scene.add_icosphere(path, radius=0.035, color=human_color, position=(0, 0, 0))
    for i_b, (a, b) in enumerate(human_bones):
        server.scene.add_spline_catmull_rom(f"/human/bone/{i_b}", points=np.array([[0, 0, 0], [0, 0, 0]]), color=human_color, line_width=6)

    joint_nodes = [
        ("shoulder", "left_shoulder_pitch_link", (0, 255, 0)),
        ("elbow", "left_elbow_pitch_link", (255, 200, 0)),
        ("wrist", "left_wrist_pitch_link", (0, 200, 255)),
        ("hand", "L_hand_base_link", (255, 100, 150)),
    ]
    for nm, _, color in joint_nodes:
        server.scene.add_icosphere(f"/robot/joints/{nm}", radius=0.025, color=color, position=(0, 0, 0))
    server.scene.add_spline_catmull_rom("/robot/left_arm_polyline", points=np.array([[0, 0, 0]]*4), color=(255, 120, 50), line_width=6)

    if CONFIG.get("show_axes_in_vis", True):
        server.scene.add_frame("/robot/base_axes", axes_length=0.12, axes_radius=0.005)
        server.scene.add_frame("/robot/tcp_axes",  axes_length=0.12, axes_radius=0.005)

    def get_active_seq():
        return seq_optimized if traj_select.value == "optimized" else seq_baseline

    def update_frame(t_idx: int):
        t_idx = int(np.clip(t_idx, 0, T - 1))
        active_seq = get_active_seq()
        q_live = active_seq["q_seq"][t_idx]
        q_dict = make_q_dict_for_urdf(urdf, robot, q_live)
        urdf_vis.update_cfg(q_dict)

        arm_pts = []
        for nm, link_nm, color in joint_nodes:
            pos = get_link_pose(robot, q_live, link_nm)[0] + ROBOT_OFFSET
            server.scene.add_icosphere(f"/robot/joints/{nm}", radius=0.025, color=color, position=pos)
            arm_pts.append(pos)
        server.scene.add_spline_catmull_rom("/robot/left_arm_polyline", points=np.array(arm_pts), color=(255, 120, 50), line_width=6)

        # 坐标轴可视化：底座轴 + TCP轴（偏移一点）
        if CONFIG.get("show_axes_in_vis", True):
            hand_pos, hand_quat_xyzw = get_link_pose(robot, q_live, "L_hand_base_link")
            server.scene.add_frame("/robot/base_axes",
                                   axes_length=0.12, axes_radius=0.005,
                                   position=hand_pos + ROBOT_OFFSET,
                                   wxyz=np.roll(hand_quat_xyzw, 1))
            R_base = R.from_quat(hand_quat_xyzw)
            R_tcp  = R_base * R_tool
            tcp_offset = np.array(CONFIG.get("tcp_axes_offset_world", [0.06, 0.0, 0.0]), dtype=float)
            server.scene.add_frame("/robot/tcp_axes",
                                   axes_length=0.12, axes_radius=0.005,
                                   position=hand_pos + ROBOT_OFFSET + tcp_offset,
                                   wxyz=np.roll(R_tcp.as_quat(), 1))

        # 人类目标（这里 left_hand 被视作 TCP点；若你打开 enable_tool_transform_in_eval，上面已换算为底座点）
        Hn = world_static_eval_tcp[t_idx] if not CONFIG["enable_tool_transform_in_eval"] else eval_targets[t_idx]
        pts_map = {
            "pelvis": Hn["pelvis"],
            "lshoulder": Hn["left_shoulder"],
            "rshoulder": Hn["right_shoulder"],
            "lelbow": Hn["left_elbow"],
            "lwrist": Hn["left_wrist"],
            "lhand": Hn["left_hand"],
        }
        for name, path in human_nodes.items():
            server.scene.add_icosphere(path, radius=0.035, color=human_color, position=pts_map[name])
        for i_b, (a, b) in enumerate(human_bones):
            server.scene.add_spline_catmull_rom(f"/human/bone/{i_b}", points=np.array([pts_map[a], pts_map[b]]), color=human_color, line_width=6)

    t0 = min(CONFIG["sample_frame_index"], T - 1)
    update_frame(t0)

    @frame_slider.on_update
    def _(ev):
        nonlocal slider_silent
        if slider_silent: return
        update_frame(int(ev.target.value))

    @traj_select.on_update
    def _(ev):
        update_frame(int(frame_slider.value))

    last_time = time.time(); accum = 0.0; cur = t0
    print("\n✅ 可视化已启动：左肩对齐 + 右肩z轴对称 + 骨盆在z轴 + 逐帧左臂 + 工具变换(方案B)")
    while True:
        now = time.time(); dt = now - last_time; last_time = now
        if play_toggle.value:
            accum += dt; step = 1.0 / max(1, int(fps_number.value))
            while accum >= step:
                accum -= step; cur += 1
                if cur >= T:
                    if loop_toggle.value: cur = 0
                    else: cur = T - 1; play_toggle.value = False; break
                update_frame(cur)
                slider_silent = True; frame_slider.value = cur; slider_silent = False
        time.sleep(0.005)

if __name__ == "__main__":
    main()