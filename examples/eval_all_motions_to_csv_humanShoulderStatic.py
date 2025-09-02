# /home/hjj/pyroki/examples/eval_all_motions_to_csv_humanShoulderStatic.py
import os
import re
import csv
import fnmatch
import time
import joblib
import torch
import numpy as np
from yourdfpy import URDF
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

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
    # 数据与模型
    "pkl_path": "/home/hjj/human2humanoid/data/new_robot/amass_for_evaluate.pkl",
    "urdf_path": "/home/hjj/human2humanoid/resources/robots/h1_2/urdf/h1_2.urdf",
    "out_dir": "/home/hjj/human2humanoid/results",

    # 评估控制
    "USE_NEXT_FRAME_TCP_FOR_IK": True,
    "auto_choose_reference": True,
    "min_neck_pelvis_dist": 0.12,
    "min_shoulder_width":  0.18,

    # 比例化与对齐策略
    "scale_human_for_eval": True,
    "scale_human_for_ik_target": False,
    "align_human_to_robot_shoulder": True,

    # IK 权重（文件名会带权重签名）
    "ik_weights": {
        "baseline":  {"pos_weight": 80.0, "ori_weight": 30.0, "elbow_pos_weight": 0.0,   "elbow_ori_weight": 0.0, "smoothness_weight": 12.0},
        "optimized": {"pos_weight": 80.0, "ori_weight": 30.0, "elbow_pos_weight": 120.0, "elbow_ori_weight": 5.0, "smoothness_weight": 12.0},
    },

    # 动作筛选与断点续传
    "resume_if_csv_exists": True,      # 已存在同名CSV则追加写入并跳过已完成的 motion_key
    "include_keys": None,              # 精确列表筛选，例如 ["0-ACCAD_xxx", "1-AMASS_yyy"]
    "include_glob": None,              # 通配符列表，例如 ["*ACCAD*", "*Walk*"]
    "exclude_glob": None,              # 排除通配，例如 ["*jog*"]
    "keys_txt_path": None,             # 文本文件，每行一个 motion_key（与 include_keys 合并）
    "max_motions": None,               # 限制数量（None=全部，筛选后再截断）
    "csv_name_suffix": "",             # 额外自定义后缀，便于区分
}

ROBOT_OFFSET = np.array([0.0, 0.0, 0.0])

# ======================== 工具函数：权重签名/文件名/目录 ========================
def num_to_tag(v: float) -> str:
    vi = int(round(v))
    if abs(v - vi) < 1e-6:
        return str(vi)
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s

def weight_signature(w: dict) -> str:
    return "pw{}_ow{}_epw{}_eow{}_sw{}".format(
        num_to_tag(w["pos_weight"]),
        num_to_tag(w["ori_weight"]),
        num_to_tag(w["elbow_pos_weight"]),
        num_to_tag(w["elbow_ori_weight"]),
        num_to_tag(w["smoothness_weight"]),
    )

def make_weights_tag(weights_cfg: dict) -> str:
    b = weight_signature(weights_cfg["baseline"])
    o = weight_signature(weights_cfg["optimized"])
    return f"b-{b}__o-{o}"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

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

# ======================== 机器人 / 姿态工具 ========================
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
            "left_hand": LH_w,
        })
    return targets

# ============== 运行一条 IK 序列（无TCP旋转/工具变换） =================
@torch.no_grad()
def run_ik_sequence(
    robot,
    frame,
    model,
    history_len,
    weights,
    use_next_frame_tcp_for_ik,
    human_static_seq_world,
):
    T = frame["left_hand_root_pos"].shape[0]
    q_prev = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    dof = robot.joints.num_actuated_joints
    q_seq = np.zeros((T, dof), dtype=float)
    ee_pos_seq = np.zeros((T, 3), dtype=float)
    chain_pos_seq = {k: np.zeros((T, 3), dtype=float) for k in ["shoulder", "elbow", "wrist", "hand"]}

    for i in range(T):
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
        target_hand_pos_w = human_static_seq_world[sel_idx]["left_hand"]

        # 朝向连续性
        _, quat_b_xyzw_prev = get_link_pose(robot, q_prev, "L_hand_base_link")
        target_wxyz_b = np.roll(R.from_quat(quat_b_xyzw_prev).as_quat(), 1)

        if weights.get("elbow_pos_weight", 0.0) == 0.0 and weights.get("elbow_ori_weight", 0.0) == 0.0:
            q_sol = pks.solve_ik(
                robot=robot, target_link_name="L_hand_base_link",
                target_position=target_hand_pos_w, target_wxyz=target_wxyz_b,
                init_q=q_prev, pos_weight=weights["pos_weight"], ori_weight=weights["ori_weight"],
                elbow_pos_weight=0.0, elbow_ori_weight=0.0, smoothness_weight=weights["smoothness_weight"],
            )
        else:
            q_sol = pks.solve_ik(
                robot=robot, target_link_name="L_hand_base_link",
                target_position=target_hand_pos_w, target_wxyz=target_wxyz_b,
                init_q=q_prev,
                target_elbow_position=pred_elbow_world_pos, target_elbow_rot_quat=pred_elbow_world_quat_wxyz,
                target_elbow_link_name="left_elbow_pitch_link",
                pos_weight=weights["pos_weight"], ori_weight=weights["ori_weight"],
                elbow_pos_weight=weights["elbow_pos_weight"], elbow_ori_weight=weights["elbow_ori_weight"],
                smoothness_weight=weights["smoothness_weight"],
            )

        if np.isnan(q_sol).any(): q_sol = q_prev.copy()
        q_prev = q_sol; q_seq[i] = q_sol

        chain_pos_seq["shoulder"][i] = get_link_pose(robot, q_sol, "left_shoulder_pitch_link")[0]
        chain_pos_seq["elbow"][i]    = get_link_pose(robot, q_sol, "left_elbow_pitch_link")[0]
        chain_pos_seq["wrist"][i]    = get_link_pose(robot, q_sol, "left_wrist_pitch_link")[0]
        chain_pos_seq["hand"][i]     = get_link_pose(robot, q_sol, "L_hand_base_link")[0]
        ee_pos_seq[i] = chain_pos_seq["hand"][i]

    return {"q_seq": q_seq, "ee_pos_seq": ee_pos_seq, "chain_pos_seq": chain_pos_seq}

# ======================== 指标 ========================
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

# ======================== 断点续传与筛选 ========================
def load_completed_keys(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    done = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "motion_key" in row and row["motion_key"]:
                done.add(row["motion_key"])
    return done

def filter_keys(all_keys, include_keys=None, include_glob=None, exclude_glob=None, keys_txt_path=None, max_motions=None):
    include_exact = set(include_keys or [])
    if keys_txt_path and os.path.exists(keys_txt_path):
        with open(keys_txt_path, "r") as f:
            for line in f:
                k = line.strip()
                if k:
                    include_exact.add(k)

    # 初步候选
    if include_exact:
        cand = [k for k in all_keys if k in include_exact]
    else:
        cand = list(all_keys)

    # 通配包含
    if include_glob:
        keep = []
        pats = list(include_glob)
        for k in cand:
            ok = any(fnmatch.fnmatch(k, pat) for pat in pats)
            if ok:
                keep.append(k)
        cand = keep if include_exact else (keep if pats else cand)

    # 通配排除
    if exclude_glob:
        pats = list(exclude_glob)
        cand = [k for k in cand if not any(fnmatch.fnmatch(k, pat) for pat in pats)]

    # 截断
    if max_motions is not None:
        cand = cand[:int(max_motions)]

    return cand

# ======================== 主流程：批量评估并导出 CSV ========================
def main():
    ensure_dir(CONFIG["out_dir"])
    all_motion = joblib.load(CONFIG["pkl_path"])
    all_keys = list(all_motion.keys())

    # 筛选
    keys = filter_keys(
        all_keys,
        include_keys=CONFIG.get("include_keys"),
        include_glob=CONFIG.get("include_glob"),
        exclude_glob=CONFIG.get("exclude_glob"),
        keys_txt_path=CONFIG.get("keys_txt_path"),
        max_motions=CONFIG.get("max_motions"),
    )
    print(f"[Select] total={len(all_keys)} -> selected={len(keys)}")

    # 载入机器人/模型
    urdf = URDF.load(CONFIG["urdf_path"], mesh_dir=os.path.dirname(CONFIG["urdf_path"]))
    robot = pk.Robot.from_urdf(urdf)
    model = ElbowNNWrapper(model_config=MODEL_INFO["model_config"])
    model.load(MODEL_INFO["path"], MODEL_INFO["stats_path"])
    history_len = MODEL_INFO["train_params"]["history_len"]

    # 机器人名义长度与肩基准
    robot_lengths = robot_nominal_upper_limb_lengths(robot)
    q_mid = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    robot_shoulder_mid = get_link_pose(robot, q_mid, "left_shoulder_pitch_link")[0]

    # 权重签名与 CSV 路径
    weights_tag = make_weights_tag(CONFIG["ik_weights"])
    #timestamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = CONFIG.get("csv_name_suffix", "")
    suffix = f"__{suffix}" if suffix else ""
    csv_name = (
        f"eval_humanShoulderStatic__{weights_tag}"
        f"__next{int(CONFIG['USE_NEXT_FRAME_TCP_FOR_IK'])}"
        f"__evalScaled{int(CONFIG['scale_human_for_eval'])}"
        f"__ikScaled{int(CONFIG['scale_human_for_ik_target'])}"
        f"{suffix}.csv"
    )
    csv_path = os.path.join(CONFIG["out_dir"], csv_name)

    # 断点续传：若存在同签名的CSV，一般我们按新文件写；但也支持用户手工指定同名文件续写
    completed = set()
    mode = "w"
    if CONFIG.get("resume_if_csv_exists", True) and os.path.exists(csv_path):
        completed = load_completed_keys(csv_path)
        mode = "a"
        print(f"[Resume] Found existing CSV, resume mode. Completed={len(completed)}")

    headers = [
        "motion_key", "T", "ref_index",
        "mse_hand_baseline", "rmse_hand_baseline",
        "mse_hand_optimized", "rmse_hand_optimized",
        "cos_ua_baseline", "cos_fa_baseline", "cos_ha_baseline", "cos_avg_baseline",
        "cos_ua_optimized", "cos_fa_optimized", "cos_ha_optimized", "cos_avg_optimized",
        "delta_cos_ua", "delta_cos_fa", "delta_cos_ha", "delta_cos_avg",
    ]

    failed = []

    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if mode == "w":
            writer.writeheader()

        pbar = tqdm(keys, desc="Evaluate motions")
        for motion_key in pbar:
            if motion_key in completed:
                pbar.set_postfix_str("skip(resume)")
                continue

            try:
                motion = all_motion[motion_key]; frame = motion["frame"]
                T = frame["left_hand_root_pos"].shape[0]
                pbar.set_postfix_str(f"T={T}")

                # 世界系原始点列
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

                # 选择参考帧
                if CONFIG.get("auto_choose_reference", True):
                    t_ref = choose_reference_index(raw_hw_list, CONFIG["min_neck_pelvis_dist"], CONFIG["min_shoulder_width"], 0)
                else:
                    t_ref = 0

                # 标准化 + 躯干锁定
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

                # 构建评估目标与 IK 目标（可独立控制比例化）
                world_static_eval = build_world_aligned_human_static(
                    human_static_seq, t_ref, robot_shoulder_mid, robot_lengths,
                    scale_to_robot=CONFIG["scale_human_for_eval"], enforce_axis_align=CONFIG["align_human_to_robot_shoulder"],
                )
                world_static_ik = build_world_aligned_human_static(
                    human_static_seq, t_ref, robot_shoulder_mid, robot_lengths,
                    scale_to_robot=CONFIG["scale_human_for_ik_target"], enforce_axis_align=CONFIG["align_human_to_robot_shoulder"],
                )

                # 时间对齐（下一帧 TCP）
                if CONFIG["USE_NEXT_FRAME_TCP_FOR_IK"]:
                    def shift_list(lst): return lst[1:] + lst[-1:]
                    world_static_eval = shift_list(world_static_eval)
                    world_static_ik   = shift_list(world_static_ik)

                # 运行 IK：baseline / optimized
                use_next = CONFIG["USE_NEXT_FRAME_TCP_FOR_IK"]
                seq_baseline = run_ik_sequence(
                    robot, frame, model, history_len,
                    CONFIG["ik_weights"]["baseline"], use_next, world_static_ik
                )
                seq_optimized = run_ik_sequence(
                    robot, frame, model, history_len,
                    CONFIG["ik_weights"]["optimized"], use_next, world_static_ik
                )

                # 评估（位置与线段方向）
                H_shoulder = np.stack([d["left_shoulder"] for d in world_static_eval], axis=0)
                H_elbow    = np.stack([d["left_elbow"] for d in world_static_eval], axis=0)
                H_wrist    = np.stack([d["left_wrist"] for d in world_static_eval], axis=0)
                H_hand     = np.stack([d["left_hand"] for d in world_static_eval], axis=0)

                mse_b, rmse_b = end_effector_mse_rmse(seq_baseline["ee_pos_seq"], H_hand)
                mse_o, rmse_o = end_effector_mse_rmse(seq_optimized["ee_pos_seq"], H_hand)

                cos_b = chain_cosine_means(seq_baseline["chain_pos_seq"], H_shoulder, H_elbow, H_wrist, H_hand)
                cos_o = chain_cosine_means(seq_optimized["chain_pos_seq"], H_shoulder, H_elbow, H_wrist, H_hand)
                cos_avg_b = float(np.mean(cos_b))
                cos_avg_o = float(np.mean(cos_o))

                row = {
                    "motion_key": motion_key,
                    "T": T,
                    "ref_index": t_ref,
                    "mse_hand_baseline": mse_b,
                    "rmse_hand_baseline": rmse_b,
                    "mse_hand_optimized": mse_o,
                    "rmse_hand_optimized": rmse_o,
                    "cos_ua_baseline": cos_b[0],
                    "cos_fa_baseline": cos_b[1],
                    "cos_ha_baseline": cos_b[2],
                    "cos_avg_baseline": cos_avg_b,
                    "cos_ua_optimized": cos_o[0],
                    "cos_fa_optimized": cos_o[1],
                    "cos_ha_optimized": cos_o[2],
                    "cos_avg_optimized": cos_avg_o,
                    "delta_cos_ua": cos_o[0] - cos_b[0],
                    "delta_cos_fa": cos_o[1] - cos_b[1],
                    "delta_cos_ha": cos_o[2] - cos_b[2],
                    "delta_cos_avg": cos_avg_o - cos_avg_b,
                }
                writer.writerow(row); f.flush()

            except Exception as e:
                failed.append((motion_key, repr(e)))
                print(f"[WARN] Failed: {motion_key} -> {e}")

    print(f"\n✅ Done. CSV saved to:\n{csv_path}")
    print(f"   weights tag: {weights_tag}")
    if failed:
        print(f"⚠ Failed {len(failed)} motions:")
        for k, msg in failed[:20]:
            print(" -", k, ":", msg)
        if len(failed) > 20:
            print(" ... total", len(failed), "failed")

if __name__ == "__main__":
    main()