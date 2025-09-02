# /home/hjj/pyroki/examples/benchmark_multi_motion_public_warmstart.py
import os
import time
import csv
import joblib
import numpy as np
import torch
from yourdfpy import URDF
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import pyroki as pk
import pyroki_snippets as pks
from NNmodel_comprehensive_exp import ElbowNNWrapper

# ======================= 配置区 =======================
MODELS_TO_BENCHMARK = [
    {"name": "Baseline IK (No NN)", "type": "baseline", "path": None, "stats_path": None, "params": {}},
    {"name": "MLP", "type": "mlp",
     "path": "/home/hjj/human2humanoid/tuned_models/hp_tune_MLP_initial_learning_rate_0.0005.pth",
     "stats_path": "/home/hjj/human2humanoid/tuned_models/hp_tune_MLP_initial_learning_rate_0.0005_stats.pkl",
     "params": {"hidden_dim": 512, "dropout_rate": 0.1, "history_len": 1}},
    {"name": "LSTM", "type": "lstm",
     "path": "/home/hjj/human2humanoid/tuned_models/hp_tune_LSTM_batch_size_256.pth",
     "stats_path": "/home/hjj/human2humanoid/tuned_models/hp_tune_LSTM_batch_size_256_stats.pkl",
     "params": {"hidden_dim": 128, "num_layers": 2, "dropout_rate": 0.2, "history_len": 1}},
    {"name": "GRU", "type": "gru",
     "path": "/home/hjj/human2humanoid/models_final_comparison/exp_GRU_Optimal.pth",
     "stats_path": "/home/hjj/human2humanoid/models_final_comparison/exp_GRU_Optimal_stats.pkl",
     "params": {"hidden_dim": 1024, "num_layers": 2, "dropout_rate": 0.7, "history_len": 1}},

    {"name": "Transformer", "type": "transformer",
     "path": "/home/hjj/human2humanoid/models_final_comparison/exp_Transformer_Optimal.pth",
     "stats_path": "/home/hjj/human2humanoid/models_final_comparison/exp_Transformer_Optimal_stats.pkl",
     "params": {"hidden_dim": 128, "num_layers": 6, "nhead": 8, "dropout_rate": 0.2, "history_len": 8}},

    {"name": "ST-Attention", "type": "st_attention",
     "path": "/home/hjj/human2humanoid/models_final_comparison_with_film/exp_STAttention_h1792_s5_lr0.00005_s5_drop2_layer2_batch128.pth",
     "stats_path": "/home/hjj/human2humanoid/models_final_comparison_with_film/exp_STAttention_h1792_s5_lr0.00005_s5_drop2_layer2_batch128_stats.pkl",
     "params": {"hidden_dim": 1792, "num_layers": 2, "nhead": 8, "dropout_rate": 0.2, "history_len": 5}},
]

CONFIG = {
    "pkl_path": "/home/hjj/human2humanoid/data/new_robot/amass_for_evaluate.pkl",
    "urdf_path": "/home/hjj/human2humanoid/resources/robots/h1_2/urdf/h1_2.urdf",

    # 多动作+多次运行
    # 填入要测试的动作 key 列表；置为 None 则默认只跑 FIRST_KEY（便于快速测试）
    "benchmark_motion_keys": [
        "0-ACCAD_Male2MartialArtsExtended_c3d_Extended 1_poses",
        "0-ACCAD_Male2MartialArtsExtended_c3d_Form 1_poses",
        "0-ACCAD_Male2MartialArtsExtended_c3d_Extended 3_poses",
        "0-ACCAD_Male2MartialArtsExtended_c3d_Extended 2_poses",
        "0-ACCAD_Female1Gestures_c3d_D3 - Conversation Gestures_poses",
        
        # 你可以在此追加更多 key，如：
        # "0-ACCAD_Female1Walking_c3d_B3 - walk1_poses",
        # "0-ACCAD_Male1Walking_c3d_Walk B16 - Walk turn change_poses",
    ],
    "num_runs_per_motion": 3,

    # 公共 warm-start 模式：off / default / baseline
    # - off: 各模型各自沿用上一帧解作初值（更接近部署）
    # - default: 每帧初值统一用 default_cfg
    # - baseline: 预先用 Baseline-IK（仅手）算一条 q_ref[t]，正式计时时所有模型用 q_ref[t-1] 作初值，且肩FK也以它为基准
    "public_warm_start_mode": "baseline",

    # IK 权重
    "ik_weights_baseline": {"pos_weight": 80.0, "ori_weight": 30.0, "smoothness_weight": 12.0},
    "ik_weights_optimized": {"pos_weight": 80.0, "ori_weight": 30.0, "elbow_pos_weight": 120.0, "elbow_ori_weight": 5.0, "smoothness_weight": 12.0},

    "robot_link_names": {
        "hand": "L_hand_base_link",
        "elbow": "left_elbow_pitch_link",
        "shoulder": "left_shoulder_pitch_link",
    },

    "WARMUP_FRAMES": 10,
    "FPS": 30.0,

    # 四元数约定
    "dataset_root_quat_order": "wxyz",  # "wxyz" 或 "xyzw"
    "pred_quat_order": "wxyz",          # "wxyz" 或 "xyzw"

    # Baseline 是否计入共享 prep（公平端到端比较）
    "include_shared_prep_for_baseline": True,

    # 统计时丢弃前 N 帧
    "skip_first_n_frames_from_stats": 0,

    # 可复现性
    "seed": 0,
    "torch_deterministic": True,

    # FK 预热次数
    "FK_PREWARM_ITERS": 5,

    # 输出
    "per_motion_csv": "inference_speed_per_motion_report.csv",
    "summary_csv": "inference_speed_cross_motion_summary.csv",
}

# ======================= 工具函数 =======================
def set_reproducible(seed=0, deterministic=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

def get_link_pose(robot, q, link_name):
    # 返回 pos(xyz) 与 quat(xyzw)
    idx = robot.links.names.index(link_name)
    pose = robot.forward_kinematics(q[None])[0, idx]
    pos = np.array(pose[4:], dtype=float)
    quat_wxyz = np.array(pose[:4], dtype=float)
    quat_xyzw = np.roll(quat_wxyz, -1)
    return pos, quat_xyzw

def shoulder_to_root(pose_local, shoulder_pose_world):
    pos_local, quat_local_xyzw = pose_local
    shoulder_pos, shoulder_quat_xyzw = shoulder_pose_world
    R_s, R_l = R.from_quat(shoulder_quat_xyzw), R.from_quat(quat_local_xyzw)
    pos_w = R_s.apply(pos_local) + shoulder_pos
    quat_w_xyzw = (R_s * R_l).as_quat()
    return pos_w, quat_w_xyzw

def normalize_quat(q, eps=1e-8):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < eps:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)  # xyzw
    return q / n

def to_xyzw(q, order):
    q = np.asarray(q, dtype=float)
    if order.lower() == "xyzw":
        return q
    elif order.lower() == "wxyz":
        return np.roll(q, -1)
    else:
        raise ValueError(f"Unknown quat order: {order}")

def to_wxyz(q, order):
    q = np.asarray(q, dtype=float)
    if order.lower() == "wxyz":
        return q
    elif order.lower() == "xyzw":
        return np.roll(q, 1)
    else        :
        raise ValueError(f"Unknown quat order: {order}")

def force_host_array(x):
    arr = np.asarray(x)
    _ = float(arr.ravel()[0]) if arr.size > 0 else 0.0
    return arr

def time_stats(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return dict(mean=np.nan, std=np.nan, p50=np.nan, p95=np.nan)
    return dict(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
    )

def build_features(traj_data, t, history_len):
    FEAT_PER_FRAME = 28  # 4*(3+4)
    x_seq = np.empty((history_len, FEAT_PER_FRAME), dtype=np.float32)
    for k in range(history_len):
        idx = max(0, t - (history_len - 1 - k))
        feat = np.concatenate([
            traj_data["left_hand_shoulder_pos"][idx],
            traj_data["left_hand_shoulder_rot"][idx],
            traj_data["right_hand_shoulder_pos"][idx],
            traj_data["right_hand_shoulder_rot"][idx],
            traj_data["left_elbow_shoulder_pos"][idx],
            traj_data["left_elbow_shoulder_rot"][idx],
            traj_data["right_elbow_shoulder_pos"][idx],
            traj_data["right_elbow_shoulder_rot"][idx],
        ], axis=0)
        x_seq[k] = feat
    next_idx = min(t + 1, len(traj_data["left_hand_shoulder_pos"]) - 1)
    x_next_tcp = np.concatenate([
        traj_data["left_hand_shoulder_pos"][next_idx],
        traj_data["left_hand_shoulder_rot"][next_idx],
        traj_data["right_hand_shoulder_pos"][next_idx],
        traj_data["right_hand_shoulder_rot"][next_idx],
    ], axis=0).astype(np.float32)
    return x_seq, x_next_tcp

def prewarm_fk(robot, q, link_name, iters=3):
    for _ in range(iters):
        _ = get_link_pose(robot, q, link_name)

def compute_public_init_q_baseline(robot, traj, default_cfg, config):
    """
    用 Baseline-IK（仅手目标）计算参考轨迹 q_ref[t]，用于公共 warm-start。
    返回：q_ref，形状 (num_frames, dof)
    """
    num_frames = int(len(traj["left_hand_root_pos"]))
    q_ref = np.empty((num_frames, default_cfg.shape[0]), dtype=float)
    q_prev = default_cfg.copy()
    for t in tqdm(range(num_frames), desc="计算公共 warm-start 参考轨迹(Baseline)"):
        target_tcp_pos = traj["left_hand_root_pos"][t]
        raw_tcp_rot = traj["left_hand_root_rot"][t]
        target_tcp_wxyz = normalize_quat(
            to_wxyz(raw_tcp_rot, config["dataset_root_quat_order"])
        )
        q_sol = pks.solve_ik(
            robot=robot,
            target_link_name=config["robot_link_names"]["hand"],
            target_position=target_tcp_pos,
            target_wxyz=target_tcp_wxyz,
            init_q=q_prev,
            **config["ik_weights_baseline"],
        )
        q_sol = force_host_array(q_sol)
        if np.isnan(q_sol).any():
            q_sol = q_prev
        q_ref[t] = q_sol
        q_prev = q_sol
    return q_ref

# ======================= 单个动作的基准执行 =======================
def benchmark_one_motion(robot, all_motion_data, motion_key, models_cfg, config, device):
    # 数据
    traj = all_motion_data[motion_key]["frame"]
    num_frames = int(len(traj["left_hand_root_pos"]))

    # 公共 warm-start 准备
    public_mode = config["public_warm_start_mode"].lower()
    default_cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    public_q_ref = None
    if public_mode == "baseline":
        public_q_ref = compute_public_init_q_baseline(robot, traj, default_cfg, config)

    # 结果
    per_model_rows = []

    for model_cfg in models_cfg:
        # ============== 预热（确保公平：IK、FK、模型都编译好） ==============
        # FK / IK 全局已预热；此处保留 per-model 的调用预热，避免首次 kernel 落在计时内
        warmup_q = default_cfg.copy()
        if model_cfg["type"] == "baseline":
            for t in range(min(config["WARMUP_FRAMES"], num_frames)):
                # FK
                _ = get_link_pose(robot, warmup_q, config["robot_link_names"]["shoulder"])
                # IK
                target_tcp_pos = traj["left_hand_root_pos"][t]
                raw_tcp_rot = traj["left_hand_root_rot"][t]
                target_tcp_wxyz = normalize_quat(
                    to_wxyz(raw_tcp_rot, config["dataset_root_quat_order"])
                )
                q_sol = pks.solve_ik(
                    robot=robot,
                    target_link_name=config["robot_link_names"]["hand"],
                    target_position=target_tcp_pos,
                    target_wxyz=target_tcp_wxyz,
                    init_q=warmup_q,
                    **config["ik_weights_baseline"],
                )
                q_sol = force_host_array(q_sol)
                if np.isnan(q_sol).any():
                    q_sol = default_cfg.copy()
                warmup_q = q_sol
        else:
            params = model_cfg["params"].copy()
            history_len = int(params.pop("history_len", 5))
            FEAT_PER_FRAME = 28; NEXT_TCP_DIM = 14
            expected_input_dim = FEAT_PER_FRAME * history_len + NEXT_TCP_DIM
            if model_cfg["type"] == "mlp":
                if "input_dim" in params and params["input_dim"] != expected_input_dim:
                    print(f"[WARN] 覆盖 MLP input_dim -> {expected_input_dim}（原 {params['input_dim']}）")
                params["input_dim"] = expected_input_dim
            model = ElbowNNWrapper(model_config={'type': model_cfg['type'], 'params': params}, device=device)
            model.load(model_cfg['path'], model_cfg['stats_path'].strip())

            for t in range(min(config["WARMUP_FRAMES"], num_frames)):
                # 肩FK：warmup 阶段也走一次
                shoulder_pos, shoulder_quat_xyzw = get_link_pose(robot, warmup_q, config["robot_link_names"]["shoulder"])
                x_seq, x_next_tcp = build_features(traj, t, history_len)
                x_seq_t = torch.from_numpy(x_seq).to(device).float()
                x_next_t = torch.from_numpy(x_next_tcp).to(device).float()
                if device.type == "cuda": torch.cuda.synchronize()
                with torch.no_grad():
                    preds = model.predict(x_seq_t, x_next_t)
                if device.type == "cuda": torch.cuda.synchronize()
                preds_np = preds.detach().cpu().numpy()
                pred_pos_s = preds_np[:3]
                pred_rot_xyzw_s = normalize_quat(to_xyzw(preds_np[3:7], config["pred_quat_order"]))
                pred_elbow_pos_w, pred_elbow_rot_xyzw_w = shoulder_to_root(
                    (pred_pos_s, pred_rot_xyzw_s), (shoulder_pos, shoulder_quat_xyzw)
                )
                elbow_wxyz = normalize_quat(to_wxyz(pred_elbow_rot_xyzw_w, "xyzw"))

                target_tcp_pos = traj["left_hand_root_pos"][t]
                raw_tcp_rot = traj["left_hand_root_rot"][t]
                target_tcp_wxyz = normalize_quat(
                    to_wxyz(raw_tcp_rot, config["dataset_root_quat_order"])
                )
                q_sol = pks.solve_ik(
                    robot=robot,
                    target_link_name=config["robot_link_names"]["hand"],
                    target_position=target_tcp_pos,
                    target_wxyz=target_tcp_wxyz,
                    target_elbow_position=pred_elbow_pos_w,
                    target_elbow_rot_quat=elbow_wxyz,
                    init_q=warmup_q,
                    **config["ik_weights_optimized"],
                )
                q_sol = force_host_array(q_sol)
                if np.isnan(q_sol).any():
                    q_sol = default_cfg.copy()
                warmup_q = q_sol

        # ============== 正式计时 ==============
        times_prep, times_infer, times_post, times_ik = [], [], [], []
        solution_previous = warmup_q.copy()  # 仍记录各模型自身的解（off 模式使用）；公共模式下仅作冗余
        nan_fallbacks = 0

        # 若是 NN，恢复模型/参数
        if model_cfg["type"] != "baseline":
            params = model_cfg["params"].copy()
            history_len = int(params.pop("history_len", 5))
            FEAT_PER_FRAME = 28; NEXT_TCP_DIM = 14
            expected_input_dim = FEAT_PER_FRAME * history_len + NEXT_TCP_DIM
            if model_cfg["type"] == "mlp":
                params["input_dim"] = expected_input_dim
            model = ElbowNNWrapper(model_config={'type': model_cfg['type'], 'params': params}, device=device)
            model.load(model_cfg['path'], model_cfg['stats_path'].strip())

        for t in tqdm(range(num_frames), desc=f"[{motion_key}] {model_cfg['name']}"):
            # 选择本帧 IK 初值（公共 warm-start 或各自上一解）
            if public_mode == "default":
                init_q_for_ik = default_cfg
            elif public_mode == "baseline" and public_q_ref is not None:
                init_q_for_ik = public_q_ref[t-1] if t > 0 else public_q_ref[0]
            else:
                init_q_for_ik = solution_previous

            # 决定肩FK所用的状态：公共 warm-start 模式下也用相同参考，以保证局部->世界变换一致
            q_for_fk = init_q_for_ik

            # prep
            if model_cfg["type"] == "baseline":
                if config["include_shared_prep_for_baseline"]:
                    t0 = time.perf_counter()
                    _ = get_link_pose(robot, q_for_fk, config["robot_link_names"]["shoulder"])
                    times_prep.append((time.perf_counter() - t0) * 1000.0)
                else:
                    times_prep.append(0.0)
                times_infer.append(0.0)
                times_post.append(0.0)

                # IK
                t_ik0 = time.perf_counter()
                target_tcp_pos = traj["left_hand_root_pos"][t]
                raw_tcp_rot = traj["left_hand_root_rot"][t]
                target_tcp_wxyz = normalize_quat(
                    to_wxyz(raw_tcp_rot, config["dataset_root_quat_order"])
                )
                q_sol = pks.solve_ik(
                    robot=robot,
                    target_link_name=config["robot_link_names"]["hand"],
                    target_position=target_tcp_pos,
                    target_wxyz=target_tcp_wxyz,
                    init_q=init_q_for_ik,
                    **config["ik_weights_baseline"],
                )
                q_sol = force_host_array(q_sol)
                if np.isnan(q_sol).any():
                    nan_fallbacks += 1
                    q_sol = init_q_for_ik
                solution_previous = q_sol
                times_ik.append((time.perf_counter() - t_ik0) * 1000.0)

            else:
                # NN 分支
                t0 = time.perf_counter()
                shoulder_pos, shoulder_quat_xyzw = get_link_pose(robot, q_for_fk, config["robot_link_names"]["shoulder"])
                x_seq, x_next_tcp = build_features(traj, t, history_len)
                x_seq_t = torch.from_numpy(x_seq).to(device).float()
                x_next_t = torch.from_numpy(x_next_tcp).to(device).float()
                times_prep.append((time.perf_counter() - t0) * 1000.0)

                if device.type == "cuda": torch.cuda.synchronize()
                t1 = time.perf_counter()
                with torch.no_grad():
                    preds = model.predict(x_seq_t, x_next_t)
                if device.type == "cuda": torch.cuda.synchronize()
                times_infer.append((time.perf_counter() - t1) * 1000.0)

                t2 = time.perf_counter()
                preds_np = preds.detach().cpu().numpy()
                pred_pos_s = preds_np[:3]
                pred_rot_xyzw_s = normalize_quat(to_xyzw(preds_np[3:7], config["pred_quat_order"]))
                pred_elbow_pos_w, pred_elbow_rot_xyzw_w = shoulder_to_root(
                    (pred_pos_s, pred_rot_xyzw_s), (shoulder_pos, shoulder_quat_xyzw)
                )
                elbow_wxyz = normalize_quat(to_wxyz(pred_elbow_rot_xyzw_w, "xyzw"))
                times_post.append((time.perf_counter() - t2) * 1000.0)

                t3 = time.perf_counter()
                target_tcp_pos = traj["left_hand_root_pos"][t]
                raw_tcp_rot = traj["left_hand_root_rot"][t]
                target_tcp_wxyz = normalize_quat(
                    to_wxyz(raw_tcp_rot, config["dataset_root_quat_order"])
                )
                q_sol = pks.solve_ik(
                    robot=robot,
                    target_link_name=config["robot_link_names"]["hand"],
                    target_position=target_tcp_pos,
                    target_wxyz=target_tcp_wxyz,
                    target_elbow_position=pred_elbow_pos_w,
                    target_elbow_rot_quat=elbow_wxyz,
                    init_q=init_q_for_ik,
                    **config["ik_weights_optimized"],
                )
                q_sol = force_host_array(q_sol)
                if np.isnan(q_sol).any():
                    nan_fallbacks += 1
                    q_sol = init_q_for_ik
                solution_previous = q_sol
                times_ik.append((time.perf_counter() - t3) * 1000.0)

        # 聚合单动作结果
        skip = int(config["skip_first_n_frames_from_stats"])
        tp = np.array(times_prep[skip:], float)
        ti = np.array(times_infer[skip:], float)
        tpost = np.array(times_post[skip:], float)
        tik = np.array(times_ik[skip:], float)
        per_frame_total = tp + ti + tpost + tik

        row = {
            "motion_key": motion_key,
            "model_name": model_cfg["name"],
            "n_frames_effective": int(len(tik)),
            "prep_time_ms": time_stats(tp)["mean"],
            "infer_time_ms": time_stats(ti)["mean"],
            "post_time_ms": time_stats(tpost)["mean"],
            "ik_time_ms": time_stats(tik)["mean"],
            "ik_time_std_ms": time_stats(tik)["std"],
            "ik_time_p95_ms": time_stats(tik)["p95"],
            "total_time_ms": float(np.mean(per_frame_total)),
            "nan_fallbacks": int(nan_fallbacks),
        }
        per_model_rows.append(row)

    return per_model_rows

# ======================= 跨动作聚合 =======================
def aggregate_results(per_motion_rows):
    """
    输入：每行包含 motion_key、model_name、n_frames_effective、各均值、ik_time_std_ms、ik_time_p95_ms
    输出：
      - macro：对每条（动作×运行）取均值的再平均
      - micro：按帧数加权的全局均值；IK std 用方差合成公式
    """
    # 按 model_name 分组
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in per_motion_rows:
        by_model[r["model_name"]].append(r)

    macro_rows, micro_rows = [], []
    for model_name, rows in by_model.items():
        # 宏：简单平均
        def avg(key): return float(np.mean([x[key] for x in rows])) if rows else np.nan
        macro_rows.append({
            "model_name": model_name,
            "macro_prep_ms": avg("prep_time_ms"),
            "macro_infer_ms": avg("infer_time_ms"),
            "macro_post_ms": avg("post_time_ms"),
            "macro_ik_ms": avg("ik_time_ms"),
            "macro_ik_p95_ms": avg("ik_time_p95_ms"),
            "macro_ik_std_ms": avg("ik_time_std_ms"),
            "macro_total_ms": avg("total_time_ms"),
            "macro_nan_fallbacks": int(np.sum([x["nan_fallbacks"] for x in rows])),
            "experiments": len(rows),
        })

        # 微：按帧数加权；IK std 合成
        N = int(np.sum([x["n_frames_effective"] for x in rows]))
        def wmean(key):
            if N == 0: return np.nan
            return float(np.sum([x[key] * x["n_frames_effective"] for x in rows]) / N)
        micro_prep = wmean("prep_time_ms")
        micro_infer = wmean("infer_time_ms")
        micro_post = wmean("post_time_ms")
        micro_ik = wmean("ik_time_ms")
        micro_total = wmean("total_time_ms")

        # 合成 IK 方差：E[X^2] - (E[X])^2
        # E[X^2] = sum_i n_i*(var_i + mean_i^2) / N
        if N > 0:
            ex2 = np.sum([
                x["n_frames_effective"] * ((x["ik_time_std_ms"] ** 2) + (x["ik_time_ms"] ** 2))
                for x in rows
            ]) / N
            micro_ik_std = float(np.sqrt(max(0.0, ex2 - micro_ik ** 2)))
        else:
            micro_ik_std = np.nan

        micro_rows.append({
            "model_name": model_name,
            "micro_prep_ms": micro_prep,
            "micro_infer_ms": micro_infer,
            "micro_post_ms": micro_post,
            "micro_ik_ms": micro_ik,
            "micro_ik_std_ms": micro_ik_std,
            "micro_total_ms": micro_total,
            "micro_frames": int(N),
            "micro_nan_fallbacks": int(np.sum([x["nan_fallbacks"] for x in rows])),
        })

    return macro_rows, micro_rows

# ======================= 主流程 =======================
def main():
    print("--- 启动【多动作×多次运行 + 公共 warm-start】基准测试（严谨版Plus） ---")
    set_reproducible(CONFIG["seed"], CONFIG["torch_deterministic"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载机器人与数据
    urdf = URDF.load(CONFIG["urdf_path"])
    robot = pk.Robot.from_urdf(urdf)
    default_cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2

    all_motion_data = joblib.load(CONFIG["pkl_path"])
    # 选择动作列表
    motion_keys = CONFIG["benchmark_motion_keys"]
    if motion_keys is None or len(motion_keys) == 0:
        # 默认取第一个键，避免一次跑太多
        motion_keys = [next(iter(all_motion_data.keys()))]
    print(f"将测试 {len(motion_keys)} 个动作键：")
    for k in motion_keys: print(" -", k)

    # 预热 IK（JAX JIT）
    print("\n--- 预热 IK (JIT 编译) ---")
    _ = pks.solve_ik(
        robot=robot,
        target_link_name=CONFIG["robot_link_names"]["hand"],
        target_position=np.array([0.5, 0.5, 0.5], dtype=float),
        target_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        init_q=default_cfg,
        **CONFIG["ik_weights_baseline"],
    )
    print("✅ IK 预热完成。")

    # 预热 FK
    print("--- 预热 FK（forward_kinematics）---")
    prewarm_fk(robot, default_cfg, CONFIG["robot_link_names"]["shoulder"], iters=CONFIG["FK_PREWARM_ITERS"])
    print("✅ FK 预热完成。")

    per_motion_rows = []
    # 多动作×多次运行
    for motion_key in motion_keys:
        print("\n" + "=" * 100)
        print(f"动作: {motion_key}")
        print("=" * 100)

        for run_idx in range(CONFIG["num_runs_per_motion"]):
            print(f"\n--- Run {run_idx+1}/{CONFIG['num_runs_per_motion']} ---")
            rows = benchmark_one_motion(robot, all_motion_data, motion_key, MODELS_TO_BENCHMARK, CONFIG, device)
            # 附加 run_idx
            for r in rows:
                r["run_idx"] = run_idx
            per_motion_rows.extend(rows)

    # 打印每动作结果（最后一轮的）
    print("\n" + "=" * 120)
    print("每动作×每次运行的结果（节选示例）")
    print("=" * 120)
    header = f"{'Motion Key':<40} | {'Run':>3} | {'Model Name':<28} | {'Prep':>7} | {'Infer':>7} | {'Post':>7} | {'IK':>8} | {'IK p95':>8} | {'IK Std':>8} | {'Total':>8} | {'Frames':>6} | {'NaN':>4}"
    print(header)
    print("-" * len(header))
    for r in per_motion_rows:
        print(f"{r['motion_key']:<40} | {r['run_idx']:>3d} | {r['model_name']:<28} | {r['prep_time_ms']:>7.3f} | {r['infer_time_ms']:>7.3f} | {r['post_time_ms']:>7.3f} | {r['ik_time_ms']:>8.3f} | {r['ik_time_p95_ms']:>8.3f} | {r['ik_time_std_ms']:>8.3f} | {r['total_time_ms']:>8.3f} | {r['n_frames_effective']:>6d} | {r['nan_fallbacks']:>4d}")

    # 聚合
    macro_rows, micro_rows = aggregate_results(per_motion_rows)

    # 打印汇总（macro + micro）
    print("\n" + "=" * 120)
    print("跨动作汇总（Macro 平均）")
    print("=" * 120)
    header_m = f"{'Model Name':<28} | {'Prep':>7} | {'Infer':>7} | {'Post':>7} | {'IK':>8} | {'IK p95':>8} | {'IK Std':>8} | {'Total':>8} | {'Runs':>5} | {'NaN Sum':>8}"
    print(header_m); print("-" * len(header_m))
    for r in sorted(macro_rows, key=lambda x: x["macro_total_ms"]):
        print(f"{r['model_name']:<28} | {r['macro_prep_ms']:>7.3f} | {r['macro_infer_ms']:>7.3f} | {r['macro_post_ms']:>7.3f} | {r['macro_ik_ms']:>8.3f} | {r['macro_ik_p95_ms']:>8.3f} | {r['macro_ik_std_ms']:>8.3f} | {r['macro_total_ms']:>8.3f} | {r['experiments']:>5d} | {r['macro_nan_fallbacks']:>8d}")

    print("\n" + "=" * 120)
    print("跨动作汇总（Micro 加权）")
    print("=" * 120)
    header_u = f"{'Model Name':<28} | {'Prep':>7} | {'Infer':>7} | {'Post':>7} | {'IK':>8} | {'IK Std':>8} | {'Total':>8} | {'Frames':>8} | {'NaN Sum':>8}"
    print(header_u); print("-" * len(header_u))
    for r in sorted(micro_rows, key=lambda x: x["micro_total_ms"]):
        print(f"{r['model_name']:<28} | {r['micro_prep_ms']:>7.3f} | {r['micro_infer_ms']:>7.3f} | {r['micro_post_ms']:>7.3f} | {r['micro_ik_ms']:>8.3f} | {r['micro_ik_std_ms']:>8.3f} | {r['micro_total_ms']:>8.3f} | {r['micro_frames']:>8d} | {r['micro_nan_fallbacks']:>8d}")

    # 保存 CSV
    if per_motion_rows:
        with open(CONFIG["per_motion_csv"], "w", newline="") as f:
            fieldnames = ["motion_key", "run_idx", "model_name", "n_frames_effective",
                          "prep_time_ms", "infer_time_ms", "post_time_ms",
                          "ik_time_ms", "ik_time_std_ms", "ik_time_p95_ms",
                          "total_time_ms", "nan_fallbacks"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_motion_rows)
        print(f"\n✅ 每动作详细结果已保存到: {CONFIG['per_motion_csv']}")

    # 汇总 CSV（macro + micro 合并）
    if macro_rows and micro_rows:
        # 合并成单表（按 model_name join）
        by_name_macro = {r["model_name"]: r for r in macro_rows}
        by_name_micro = {r["model_name"]: r for r in micro_rows}
        merged = []
        for name in by_name_macro.keys():
            m = by_name_macro[name]; u = by_name_micro.get(name, {})
            merged.append({
                "model_name": name,
                # macro
                "macro_prep_ms": m["macro_prep_ms"], "macro_infer_ms": m["macro_infer_ms"], "macro_post_ms": m["macro_post_ms"],
                "macro_ik_ms": m["macro_ik_ms"], "macro_ik_p95_ms": m["macro_ik_p95_ms"], "macro_ik_std_ms": m["macro_ik_std_ms"],
                "macro_total_ms": m["macro_total_ms"], "macro_experiments": m["experiments"], "macro_nan_sum": m["macro_nan_fallbacks"],
                # micro
                "micro_prep_ms": u.get("micro_prep_ms", np.nan), "micro_infer_ms": u.get("micro_infer_ms", np.nan), "micro_post_ms": u.get("micro_post_ms", np.nan),
                "micro_ik_ms": u.get("micro_ik_ms", np.nan), "micro_ik_std_ms": u.get("micro_ik_std_ms", np.nan),
                "micro_total_ms": u.get("micro_total_ms", np.nan), "micro_frames": u.get("micro_frames", 0), "micro_nan_sum": u.get("micro_nan_fallbacks", 0),
            })
        with open(CONFIG["summary_csv"], "w", newline="") as f:
            fieldnames = list(merged[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged)
        print(f"✅ 跨动作汇总已保存到: {CONFIG['summary_csv']}")

    print("\n提示：")
    print("- 使用 public_warm_start_mode='baseline' 可最大程度对齐“上一帧初值与肩FK参考”，排序更稳。")
    print("- macro 更抗极值；micro 更接近总体实时成本（按帧数加权）。")
    print("- 如需仅比较 IK，不关心 prep/infer/post，可将 include_shared_prep_for_baseline=False。")

if __name__ == "__main__":
    main()