# /home/hjj/pyroki/examples/NNmodel_comprehensive_exp.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import joblib
import os
import random
import time
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from collections import defaultdict

# [依赖] Mamba架构需要此库
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

# ===================================================================
# ====================== 0. 可复现性配置 ==========================
# ===================================================================
def set_seed(seed_value=42):
    """设置全局随机种子以确保实验的可复现性。"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ===================================================================
# ====================== 1. 数据集定义 ============================
# ===================================================================
class HandElbowTrajectoryDataset(Dataset):
    def __init__(self, samples_list, model_type='mlp', history_len=5):
        self.samples = samples_list; self.model_type = model_type; self.history_len = history_len
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]; x_seq_full = torch.from_numpy(sample["input"]["seq"]); x_seq = x_seq_full[-self.history_len:]; x_next_tcp = torch.from_numpy(sample["input"]["next_tcp"]); y_target = torch.from_numpy(sample["target"])
        if 'mlp' in self.model_type: return torch.cat([x_seq.flatten(), x_next_tcp], dim=0), y_target
        else: return {"seq": x_seq, "next_tcp": x_next_tcp}, y_target

# ===================================================================
# ====================== 2. 网络架构定义 ==========================
# ===================================================================
# (所有模型定义与您提供的版本完全相同，此处省略以保持简洁)
class ElbowMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, output_dim=14):
        super().__init__(); self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, output_dim))
    def forward(self, x): return self.model(x)
class ElbowLSTM(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout_rate, input_dim=28, tcp_dim=14, output_dim=14):
        super().__init__(); self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate); self.fc = nn.Sequential(nn.Linear(hidden_dim + tcp_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x_seq, x_next_tcp):
        lstm_out, _ = self.lstm(x_seq); last_features = lstm_out[:, -1, :]; combined = torch.cat([last_features, x_next_tcp], dim=1); return self.fc(combined)
class ElbowGRU(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout_rate, input_dim=28, tcp_dim=14, output_dim=14):
        super().__init__(); self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate); self.fc = nn.Sequential(nn.Linear(hidden_dim + tcp_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x_seq, x_next_tcp):
        gru_out, _ = self.gru(x_seq); last_features = gru_out[:, -1, :]; combined = torch.cat([last_features, x_next_tcp], dim=1); return self.fc(combined)
class GoalAttentiveGRU(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout_rate, nhead=8, input_dim=28, tcp_dim=14, output_dim=14):
        super().__init__(); self.history_encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate); self.goal_encoder = nn.Sequential(nn.Linear(tcp_dim, hidden_dim), nn.ReLU()); self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True); self.prediction_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x_seq, x_next_tcp):
        history_features, _ = self.history_encoder(x_seq); goal_feature = self.goal_encoder(x_next_tcp); query = goal_feature.unsqueeze(1); attended_output, _ = self.attention(query, history_features, history_features); final_feature = attended_output.squeeze(1); return self.prediction_head(final_feature)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__(); self.dropout = nn.Dropout(p=dropout); position = torch.arange(max_len).unsqueeze(1); div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)); pe = torch.zeros(max_len, 1, d_model); pe[:, 0, 0::2] = torch.sin(position * div_term); pe[:, 0, 1::2] = torch.cos(position * div_term); self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)].transpose(0, 1); return self.dropout(x)
class ElbowTransformer(nn.Module):
    def __init__(self, hidden_dim, num_layers, nhead, dropout_rate, input_dim=28, tcp_dim=14, output_dim=14):
        super().__init__(); self.d_model = hidden_dim; self.input_proj = nn.Linear(input_dim, self.d_model); self.goal_proj = nn.Linear(tcp_dim, self.d_model); self.pos_encoder = PositionalEncoding(self.d_model, dropout_rate); encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=self.d_model*4, dropout=dropout_rate, batch_first=True); self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers); decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=self.d_model*4, dropout=dropout_rate, batch_first=True); self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers); self.output_head = nn.Linear(self.d_model, output_dim)
    def forward(self, x_seq, x_next_tcp):
        src = self.input_proj(x_seq) * math.sqrt(self.d_model); src = self.pos_encoder(src); memory = self.transformer_encoder(src); tgt = self.goal_proj(x_next_tcp).unsqueeze(1); output = self.transformer_decoder(tgt, memory); prediction = self.output_head(output.squeeze(1)); return prediction
class ElbowRTransformer(nn.Module):
    def __init__(self, hidden_dim, num_layers_transformer, num_layers_gru, nhead, dropout_rate, input_dim=28, tcp_dim=14, output_dim=14):
        super().__init__(); self.d_model = hidden_dim; self.recurrent_preprocessor = nn.GRU(input_dim, hidden_dim, num_layers_gru, batch_first=True, dropout=dropout_rate); self.goal_proj = nn.Linear(tcp_dim, self.d_model); self.pos_encoder = PositionalEncoding(self.d_model, dropout_rate); encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=self.d_model*4, dropout=dropout_rate, batch_first=True); self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_transformer); decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=self.d_model*4, dropout=dropout_rate, batch_first=True); self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers_transformer); self.output_head = nn.Linear(self.d_model, output_dim)
    def forward(self, x_seq, x_next_tcp):
        recurrent_features, _ = self.recurrent_preprocessor(x_seq); src = recurrent_features * math.sqrt(self.d_model); src = self.pos_encoder(src); memory = self.transformer_encoder(src); tgt = self.goal_proj(x_next_tcp).unsqueeze(1); output = self.transformer_decoder(tgt, memory); prediction = self.output_head(output.squeeze(1)); return prediction
class ElbowGatedFusion(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout_rate, input_dim=28, tcp_dim=14, output_dim=14):
        super().__init__(); self.hidden_dim = hidden_dim; self.history_encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate); self.gate_generator = nn.Sequential(nn.Linear(tcp_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim * 2)); self.prediction_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, output_dim))
    def forward(self, x_seq, x_next_tcp):
        gru_out, _ = self.history_encoder(x_seq); history_feature = gru_out[:, -1, :]; gate_signals = self.gate_generator(x_next_tcp); gamma = gate_signals[:, :self.hidden_dim]; beta = gate_signals[:, self.hidden_dim:]; modulated_feature = history_feature * gamma + beta; return self.prediction_head(modulated_feature)
class ElbowCoAttention(nn.Module):
    def __init__(self, hidden_dim, num_layers, nhead, dropout_rate, input_dim=28, tcp_dim=14, output_dim=14):
        super().__init__(); self.history_encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True); self.goal_encoder = nn.Linear(tcp_dim, hidden_dim); self.goal_to_history_attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True, dropout=dropout_rate); self.history_to_goal_attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True, dropout=dropout_rate); self.prediction_head = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim * 2, output_dim))
    def forward(self, x_seq, x_next_tcp):
        history_features, _ = self.history_encoder(x_seq); history_summary = history_features[:, -1, :]; goal_feature = self.goal_encoder(x_next_tcp); goal_qkv = goal_feature.unsqueeze(1); attn_h, _ = self.goal_to_history_attn(goal_qkv, history_features, history_features); attn_h = attn_h.squeeze(1); attn_g, _ = self.history_to_goal_attn(history_summary.unsqueeze(1), goal_qkv, goal_qkv); attn_g = attn_g.squeeze(1); combined = torch.cat([history_summary, goal_feature, attn_h, attn_g], dim=1); return self.prediction_head(combined)
class ElbowHyperNetwork(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout_rate, input_dim=28, tcp_dim=14, output_dim=14):
        super().__init__(); self.history_encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate); head_layer1_out = hidden_dim // 2; hypernet_out_size = (hidden_dim * head_layer1_out + head_layer1_out) + (head_layer1_out * output_dim + output_dim); self.hypernet = nn.Sequential(nn.Linear(tcp_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hypernet_out_size)); self.w1_shape, self.b1_len = (hidden_dim, head_layer1_out), head_layer1_out; self.w2_shape, self.b2_len = (head_layer1_out, output_dim), output_dim
    def forward(self, x_seq, x_next_tcp):
        gru_out, _ = self.history_encoder(x_seq); history_feature = gru_out[:, -1, :]; params = self.hypernet(x_next_tcp); b, _ = history_feature.shape; w1_end = np.prod(self.w1_shape); b1_end = w1_end + self.b1_len; w2_end = b1_end + np.prod(self.w2_shape); b2_end = w2_end + self.b2_len; w1 = params[:, :w1_end].view(b, *self.w1_shape); b1 = params[:, w1_end:b1_end]; w2 = params[:, b1_end:w2_end].view(b, *self.w2_shape); b2 = params[:, w2_end:b2_end]; x = torch.bmm(history_feature.unsqueeze(1), w1).squeeze(1) + b1; x = F.relu(x); x = torch.bmm(x.unsqueeze(1), w2).squeeze(1) + b2; return x
class ElbowSTAttention(nn.Module):
    def __init__(self, hidden_dim, num_layers, nhead, dropout_rate, input_dim=28, tcp_dim=14, output_dim=14):
        super().__init__(); self.d_model = hidden_dim; self.joint_dim = 7; self.num_joints = input_dim // self.joint_dim; self.temporal_encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate); num_attention_heads = 1 if self.joint_dim % nhead != 0 else nhead; self.spatial_attention = nn.MultiheadAttention(self.joint_dim, num_heads=num_attention_heads, batch_first=True, dropout=dropout_rate); self.goal_modulator = nn.Sequential(nn.Linear(tcp_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim * 2)); fusion_input_dim = hidden_dim + self.num_joints * self.joint_dim; self.fusion = nn.Sequential(nn.Linear(fusion_input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, output_dim))
    def forward(self, x_seq, x_next_tcp):
        batch_size, seq_len, input_dim = x_seq.shape; temporal_features, _ = self.temporal_encoder(x_seq); temporal_summary = temporal_features[:, -1, :]; last_frame = x_seq[:, -1, :]; expected_dim = self.num_joints * self.joint_dim
        if input_dim != expected_dim:
            if input_dim < expected_dim: padding = torch.zeros(batch_size, expected_dim - input_dim, device=last_frame.device); last_frame = torch.cat([last_frame, padding], dim=1)
            else: last_frame = last_frame[:, :expected_dim]
        joints = last_frame.reshape(batch_size, self.num_joints, self.joint_dim); spatial_attended, _ = self.spatial_attention(joints, joints, joints); spatial_summary = spatial_attended.reshape(batch_size, -1); goal_gates = self.goal_modulator(x_next_tcp); gate_alpha = torch.sigmoid(goal_gates[:, :self.d_model]); gate_beta = goal_gates[:, self.d_model:]; modulated_temporal = temporal_summary * gate_alpha + gate_beta; combined_features = torch.cat([modulated_temporal, spatial_summary], dim=1); return self.fusion(combined_features)

# ===================================================================
# ====================== 3. 训练器封装 ============================
# ===================================================================
class ElbowNNWrapper:
    # (ElbowNNWrapper类的所有方法与您提供的版本完全相同，此处省略)
    def __init__(self, model_config, lr=1e-3, device='cuda'):
        self.model_type = model_config['type']; self.device = device; params = model_config['params'].copy()
        if self.model_type == 'mlp': params.pop('num_layers', None); params.pop('nhead', None); self.model = ElbowMLP(**params).to(device)
        elif self.model_type == 'lstm': params.pop('nhead', None); self.model = ElbowLSTM(**params).to(device)
        elif self.model_type == 'gru': params.pop('nhead', None); self.model = ElbowGRU(**params).to(device)
        elif self.model_type == 'goal_attn_gru': self.model = GoalAttentiveGRU(**params).to(device)
        elif self.model_type == 'transformer': self.model = ElbowTransformer(**params).to(device)
        elif self.model_type == 'r_transformer': self.model = ElbowRTransformer(**params).to(device)
        elif self.model_type == 'gated_fusion': params.pop('nhead', None); self.model = ElbowGatedFusion(**params).to(device)
        elif self.model_type == 'co_attention': self.model = ElbowCoAttention(**params).to(device)
        elif self.model_type == 'hypernetwork': params.pop('nhead', None); self.model = ElbowHyperNetwork(**params).to(device)
        elif self.model_type == 'st_attention': self.model = ElbowSTAttention(**params).to(device)
        else: raise ValueError(f"不支持的模型类型: {self.model_type}")
        self.criterion = nn.MSELoss(); self.optimizer = optim.Adam(self.model.parameters(), lr=lr); self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.2, patience=10, verbose=False); self.stats = None
    def load(self, model_path, stats_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True)); self.model.eval()
        print(f"✅ 模型权重已从 {model_path} 加载。"); self.stats = joblib.load(stats_path); print(f"📊 归一化数据已从 {stats_path} 加载。")
    def predict(self, x_seq, x_next_tcp=None):
        self.model.eval()
        with torch.no_grad():
            if 'mlp' in self.model_type:
                if x_seq.dim() > 1: x_input = torch.cat([x_seq.flatten(), x_next_tcp], dim=0)
                else: x_input = x_seq
                x_input = x_input.to(self.device); history_len = x_seq.shape[0] if x_seq.dim() > 1 else (x_input.shape[0] - 14) // 28
                pred = self.model(self._normalize_mlp(x_input, history_len)); return pred
            else:
                x_seq = x_seq.to(self.device); x_next_tcp = x_next_tcp.to(self.device)
                if x_seq.dim() == 2: x_seq = x_seq.unsqueeze(0)
                if x_next_tcp.dim() == 1: x_next_tcp = x_next_tcp.unsqueeze(0)
                x_seq_norm, x_tcp_norm = self._normalize_rnn(x_seq, x_next_tcp); pred = self.model(x_seq_norm, x_tcp_norm); return pred.squeeze(0)
    def train(self, train_dataset, val_dataset, model_save_path, stats_path, epochs=200, batch_size=512, patience=30):
        self.stats = joblib.load(stats_path)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        best_val_loss = float('inf'); patience_counter = 0
        for epoch in range(1, epochs + 1):
            self.model.train(); total_train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Train E{epoch:03d}", leave=False)
            for x_batch, y_batch in pbar:
                y_batch = y_batch.to(self.device, non_blocking=True)
                if 'mlp' in self.model_type:
                    x_batch = x_batch.to(self.device, non_blocking=True); pred = self.model(self._normalize_mlp(x_batch, train_dataset.history_len)); loss = self.criterion(pred, y_batch)
                else:
                    x_seq, x_tcp = x_batch["seq"].to(self.device), x_batch["next_tcp"].to(self.device); x_seq_norm, x_tcp_norm = self._normalize_rnn(x_seq, x_tcp)
                    pred = self.model(x_seq_norm, x_tcp_norm); loss = self.criterion(pred, y_batch)
                self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0); self.optimizer.step(); total_train_loss += loss.item()
                pbar.set_postfix_str(f"loss: {loss.item():.6f}")
            avg_train_loss = total_train_loss / len(train_loader)
            self.model.eval(); total_val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    y_batch = y_batch.to(self.device, non_blocking=True)
                    if 'mlp' in self.model_type:
                        x_batch = x_batch.to(self.device, non_blocking=True); pred = self.model(self._normalize_mlp(x_batch, val_dataset.history_len)); loss = self.criterion(pred, y_batch)
                    else:
                        x_seq, x_tcp = x_batch["seq"].to(self.device), x_batch["next_tcp"].to(self.device); x_seq_norm, x_tcp_norm = self._normalize_rnn(x_seq, x_tcp)
                        pred = self.model(x_seq_norm, x_tcp_norm); loss = self.criterion(pred, y_batch)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader); self.scheduler.step(avg_val_loss)
            wandb.log({"epoch": epoch, "avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss, "learning_rate": self.optimizer.param_groups[0]['lr']})
            print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss; patience_counter = 0; torch.save(self.model.state_dict(), model_save_path)
                print(f"  -> New best model saved! Val Loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
            if patience_counter >= patience: print(f"🛑 Early stopping triggered. Best Val_loss: {best_val_loss:.6f}"); break
    def _normalize_mlp(self, x, history_len):
        key = f'flat_input_h{history_len}'; 
        try: mean = torch.from_numpy(self.stats['mean'][key]).to(self.device); std = torch.from_numpy(self.stats['std'][key]).to(self.device)
        except KeyError: mean = torch.from_numpy(self.stats['mean']['flat_input']).to(self.device); std = torch.from_numpy(self.stats['std']['flat_input']).to(self.device)
        return (x - mean) / std
    def _normalize_rnn(self, x_seq, x_tcp):
        history_len = x_seq.shape[1]
        try:
            mean_seq = torch.from_numpy(self.stats['sequential']['mean_seq'][-history_len:]).to(self.device); std_seq = torch.from_numpy(self.stats['sequential']['std_seq'][-history_len:]).to(self.device)
            mean_tcp = torch.from_numpy(self.stats['sequential']['mean_tcp']).to(self.device); std_tcp = torch.from_numpy(self.stats['sequential']['std_tcp']).to(self.device)
        except KeyError:
            mean_seq = torch.from_numpy(self.stats['mean']['seq'][-history_len:]).to(self.device); std_seq = torch.from_numpy(self.stats['std']['seq'][-history_len:]).to(self.device)
            mean_tcp = torch.from_numpy(self.stats['mean']['next_tcp']).to(self.device); std_tcp = torch.from_numpy(self.stats['std']['next_tcp']).to(self.device)
        return (x_seq - mean_seq) / std_seq, (x_tcp - mean_tcp) / std_tcp

    # [核心修正] 确保预处理函数会保存 motion_key
    @staticmethod
    def preprocess_from_pkl(pkl_path, save_path):
        data = joblib.load(pkl_path); all_samples = []
        full_history_len = 12
        for key, motion_data in tqdm(data.items(), desc="预处理数据"):
            if "frame" not in motion_data or "left_hand_shoulder_pos" not in motion_data["frame"]: continue
            frames = motion_data["frame"]; num_frames = frames["left_hand_shoulder_pos"].shape[0]
            for t in range(num_frames):
                seq_feats = []
                for k in range(full_history_len - 1, -1, -1):
                    idx = max(0, t - k)
                    feat = np.concatenate([ frames["left_hand_shoulder_pos"][idx], frames["left_hand_shoulder_rot"][idx], frames["right_hand_shoulder_pos"][idx], frames["right_hand_shoulder_rot"][idx], frames["left_elbow_shoulder_pos"][idx], frames["left_elbow_shoulder_rot"][idx], frames["right_elbow_shoulder_pos"][idx], frames["right_elbow_shoulder_rot"][idx] ])
                    seq_feats.append(feat)
                input_seq = np.stack(seq_feats, axis=0)
                next_idx = min(t + 1, num_frames - 1)
                next_tcp = np.concatenate([ frames["left_hand_shoulder_pos"][next_idx], frames["left_hand_shoulder_rot"][next_idx], frames["right_hand_shoulder_pos"][next_idx], frames["right_hand_shoulder_rot"][next_idx] ])
                target_elbow = np.concatenate([ frames["left_elbow_shoulder_pos"][next_idx], frames["left_elbow_shoulder_rot"][next_idx], frames["right_elbow_shoulder_pos"][next_idx], frames["right_elbow_shoulder_rot"][next_idx] ])
                
                # [新增] 将 motion_key 保存到每个样本中
                sample = {"input": {"seq": input_seq.astype(np.float32), "next_tcp": next_tcp.astype(np.float32)}, "target": target_elbow.astype(np.float32), "motion_key": key}
                all_samples.append(sample)
                
        print("\n正在计算归一化统计数据...")
        all_seqs = np.stack([s['input']['seq'] for s in all_samples]); all_tcps = np.stack([s['input']['next_tcp'] for s in all_samples])
        stats = {'sequential': {}, 'mean': {}, 'std': {}}
        for history_len in [1, 3, 5, 8, 12, 16, 32, 64, 128]:
            flat_input = np.concatenate([all_seqs[:, -history_len:, :].reshape(len(all_samples), -1), all_tcps], axis=1)
            mean_flat, std_flat = flat_input.mean(axis=0), flat_input.std(axis=0)
            std_flat[std_flat < 1e-6] = 1.0; stats['mean'][f'flat_input_h{history_len}'] = mean_flat; stats['std'][f'flat_input_h{history_len}'] = std_flat
        mean_seq, std_seq = all_seqs.mean(axis=0), all_seqs.std(axis=0); mean_tcp, std_tcp = all_tcps.mean(axis=0), all_tcps.std(axis=0)
        std_seq[std_seq < 1e-6] = 1.0; std_tcp[std_tcp < 1e-6] = 1.0
        stats['sequential']['mean_seq'] = mean_seq; stats['sequential']['std_seq'] = std_seq; stats['sequential']['mean_tcp'] = mean_tcp; stats['sequential']['std_tcp'] = std_tcp
        stats_save_path = save_path.replace('.pt', '_stats.pkl')
        joblib.dump(stats, stats_save_path); print(f"归一化数据已保存到 {stats_save_path}")
        torch.save(all_samples, save_path); print(f"{len(all_samples)} 个训练样本已保存到 {save_path}")

# ===================================================================
# ====================== 5. 实验运行主逻辑 (修正版) =================
# ===================================================================
if __name__ == "__main__":
    set_seed(42)
    
    # --- 文件路径配置 ---
    pkl_path = '/home/hjj/human2humanoid/data/new_robot/amass_all_corrected_new_full_final.pkl'
    # 建议为新的预处理数据使用新文件名，以避免混淆
    preprocessed_save_path = '/home/hjj/human2humanoid/hand_elbow_training_data_full_with_keys.pt' 

    # --- 数据预处理 ---
    # 第一次运行时设为True，之后可以设为False以跳过
    run_preprocessing = True
    if run_preprocessing or not os.path.exists(preprocessed_save_path):
        print("--- [步骤 1] 运行数据预处理 (将为每个样本添加motion_key)... ---")
        ElbowNNWrapper.preprocess_from_pkl(pkl_path=pkl_path, save_path=preprocessed_save_path)
    
    # --- [核心修正] 按动作序列划分训练集和验证集 (防止数据泄漏) ---
    print("\n--- [步骤 2] 按动作序列划分训练集和验证集 ---")
    
    # 1. 加载包含 motion_key 的预处理数据
    all_samples_data = torch.load(preprocessed_save_path)
    
    # 2. 创建一个从 motion_key 到其所有样本的映射
    motion_key_to_samples = defaultdict(list)
    for sample in tqdm(all_samples_data, desc="按动作key组织样本"):
        # 确保 sample 字典中有 'motion_key'
        if "motion_key" not in sample:
            raise ValueError("错误: 预处理后的样本中缺少 'motion_key'。请修改 preprocess_from_pkl 函数。")
        motion_key_to_samples[sample["motion_key"]].append(sample)

    # 3. 获取所有唯一的动作key，并随机打乱
    all_motion_keys = list(motion_key_to_samples.keys())
    random.shuffle(all_motion_keys)
    
    # 4. 按 90/10 的比例划分动作key
    val_split_size = int(len(all_motion_keys) * 0.1) # 10%的动作作为验证集
    val_keys = set(all_motion_keys[:val_split_size])
    train_keys = set(all_motion_keys[val_split_size:])
    
    print(f"动作总数: {len(all_motion_keys)}, 训练动作数: {len(train_keys)}, 验证动作数: {len(val_keys)}")

    # 5. 根据划分好的keys，创建最终的训练和验证样本列表
    train_samples = []
    for key in train_keys:
        train_samples.extend(motion_key_to_samples[key])
        
    val_samples = []
    for key in val_keys:
        val_samples.extend(motion_key_to_samples[key])

    print(f"数据划分完成: {len(train_samples)} 训练样本, {len(val_samples)} 验证样本")
    
    # --- 实验配置与运行 ---
    print("\n--- [步骤 3] 开始模型训练实验 ---")
    EXPERIMENT_CONFIGS = [
        {"run_name": "MLP_no_leak", "model_config": { "type": "mlp", "params": {"hidden_dim": 512, "dropout_rate": 0.1}}, "train_params": {"lr": 5e-4, "batch_size": 512, "history_len": 1}},
        {"run_name": "LSTM_no_leak", "model_config": { "type": "lstm", "params": {"hidden_dim": 128, "num_layers": 2, "dropout_rate": 0.2}}, "train_params": {"lr": 1e-3, "batch_size": 256, "history_len": 1}},
        {"run_name": "GRU_no_leak", "model_config": { "type": "gru", "params": {"hidden_dim": 1024, "num_layers": 2, "dropout_rate": 0.7}}, "train_params": {"lr": 1e-3, "batch_size": 512, "history_len": 1}},
        {"run_name": "Transformer_no_leak", "model_config": { "type": "transformer", "params": {"hidden_dim": 128, "num_layers": 6, "nhead": 8, "dropout_rate": 0.2}}, "train_params": {"lr": 1e-3, "batch_size": 128, "history_len": 8}},      
        {"run_name": "STAttention_no_leak", "model_config": { "type": "st_attention", "params": {"hidden_dim": 1792, "num_layers": 2, "nhead": 8, "dropout_rate": 0.2} }, "train_params": {"lr": 5e-5, "batch_size": 128, "history_len": 5}},
    ]

    for exp_config in EXPERIMENT_CONFIGS:
        print(f"\n{'='*25} 开始实验: {exp_config['run_name']} {'='*25}")
        
        wandb.init(project="Model_exploration_no_leak", name=exp_config['run_name'], reinit=True)
        
        model_type = exp_config['model_config']['type']
        history_len = exp_config['train_params']['history_len']
        train_dataset = HandElbowTrajectoryDataset(train_samples, model_type=model_type, history_len=history_len)
        val_dataset = HandElbowTrajectoryDataset(val_samples, model_type=model_type, history_len=history_len)
        
        model_save_dir = "/home/hjj/human2humanoid/models_final_comparison_no_leak"
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f"{exp_config['run_name']}.pth")
        
        if 'mlp' in model_type:
            exp_config['model_config']['params']['input_dim'] = (history_len * 28) + 14
    
        trainer = ElbowNNWrapper(model_config=exp_config['model_config'], lr=exp_config['train_params']['lr'])
        
        trainer.train(
            train_dataset, val_dataset, 
            model_save_path=model_save_path, 
            stats_path=preprocessed_save_path.replace('.pt', '_stats.pkl'), 
            batch_size=exp_config['train_params']['batch_size'], 
            patience=30, epochs=200
        )

        stats_source_path = preprocessed_save_path.replace('.pt', '_stats.pkl')
        stats_dest_path = model_save_path.replace('.pth', '_stats.pkl')
        if os.path.exists(stats_source_path):
            os.system(f'cp "{stats_source_path}" "{stats_dest_path}"')
            print(f"Copied stats to {stats_dest_path}")
        
        wandb.finish()

    print("\n✅ 所有对比实验完成!")