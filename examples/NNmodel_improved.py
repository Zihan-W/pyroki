import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import os
import random

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

class HandElbowTrajectoryDataset(Dataset):
    def __init__(self, samples_list):
        self.samples = samples_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x_seq = torch.from_numpy(sample["input"]["seq"])
        x_next_tcp = torch.from_numpy(sample["input"]["next_tcp"])
        x_input = torch.cat([x_seq.flatten(), x_next_tcp], dim=0)
        y_target = torch.from_numpy(sample["target"])
        return x_input, y_target

class ElbowMLP(nn.Module):
    def __init__(self, input_dim=154, hidden_dim=256, output_dim=14):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class ElbowNNWrapper:
    def __init__(self, input_dim=154, hidden_dim=256, output_dim=14, lr=1e-3, device='cuda'):
        self.model = ElbowMLP(input_dim, hidden_dim, output_dim).to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        self.stats = None

    def train(self, train_dataset, val_dataset, epochs=10000, batch_size=256, patience=30):
        stats_path = '/home/hjj/human2humanoid/hand_elbow_training_data_stats.pkl'
        try:
            self.stats = joblib.load(stats_path)
            self.mean = torch.from_numpy(self.stats['mean']).to(self.device)
            self.std = torch.from_numpy(self.stats['std']).to(self.device)
        except FileNotFoundError:
            print(f"错误: 找不到归一化文件 {stats_path}。请先运行预处理。")
            return

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        best_val_loss = float('inf'); patience_counter = 0; best_epoch = 0

        for epoch in range(1, epochs + 1):
            self.model.train(); total_train_loss = 0.0
            pbar = tqdm(train_dataloader, desc=f"Train E{epoch:4d}", leave=False)
            for x_batch, y_batch in pbar:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                x_batch = (x_batch - self.mean) / self.std
                pred = self.model(x_batch)
                loss = self.criterion(pred, y_batch)
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
                total_train_loss += loss.item()
                pbar.set_postfix({"train_loss": f"{loss.item():.6f}"})
            avg_train_loss = total_train_loss / len(train_dataloader)

            self.model.eval(); total_val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_dataloader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    x_batch = (x_batch - self.mean) / self.std
                    pred = self.model(x_batch)
                    loss = self.criterion(pred, y_batch)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_dataloader)

            self.scheduler.step(avg_val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            wandb.log({"epoch": epoch, "avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss, "learning_rate": current_lr})
            print(f"📘 Epoch {epoch:4d} | Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.1e}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss; patience_counter = 0; best_epoch = epoch
                torch.save(self.model.state_dict(), "best_model.pth")
                print(f"  ✨ New best model saved with validation loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered. Best model from epoch {best_epoch}"); break
    
    def save(self, path): torch.save(self.model.state_dict(), path)
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()
        stats_path = path.replace('.pth', '_stats.pkl')
        try:
            self.stats = joblib.load(stats_path)
            self.mean = torch.from_numpy(self.stats['mean']).to(self.device)
            self.std = torch.from_numpy(self.stats['std']).to(self.device)
            print(f"📊 Normalization stats loaded from {stats_path}")
        except FileNotFoundError:
            self.stats = None; print(f"⚠️ Warning: Stats file not found at {stats_path}.")

    def predict(self, x_input: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            x_input = x_input.to(self.device)
            if self.stats: x_input = (x_input - self.mean) / self.std
            if x_input.dim() == 1: 
                x_input = x_input.unsqueeze(0)
                return self.model(x_input)[0]
            return self.model(x_input)

    @staticmethod
    def preprocess_from_pkl(pkl_path, save_path):
        data = joblib.load(pkl_path)
        print(f"✅ Loaded {len(data)} trajectories from {pkl_path}")
        all_samples = []
        for motion_key, motion_data in tqdm(data.items(), desc="Processing trajectories"):
            # [修正] 将错误的一行代码拆分为正确的两行
            frames = motion_data["frame"]
            num_frames = frames["left_hand_shoulder_pos"].shape[0]

            for t in range(num_frames):
                seq_feats = []
                for k in range(4, -1, -1):
                    idx = max(0, t - k)
                    feat = np.concatenate([ frames["left_hand_shoulder_pos"][idx], frames["left_hand_shoulder_rot"][idx], frames["right_hand_shoulder_pos"][idx], frames["right_hand_shoulder_rot"][idx], frames["left_elbow_shoulder_pos"][idx], frames["left_elbow_shoulder_rot"][idx], frames["right_elbow_shoulder_pos"][idx], frames["right_elbow_shoulder_rot"][idx], ])
                    seq_feats.append(feat)
                input_seq = np.stack(seq_feats, axis=0)
                next_idx = min(t + 1, num_frames - 1)
                next_tcp = np.concatenate([ frames["left_hand_shoulder_pos"][next_idx], frames["left_hand_shoulder_rot"][next_idx], frames["right_hand_shoulder_pos"][next_idx], frames["right_hand_shoulder_rot"][next_idx], ])
                target_elbow = np.concatenate([ frames["left_elbow_shoulder_pos"][next_idx], frames["left_elbow_shoulder_rot"][next_idx], frames["right_elbow_shoulder_pos"][next_idx], frames["right_elbow_shoulder_rot"][next_idx], ])
                sample = {"input": {"seq": input_seq.astype(np.float32), "next_tcp": next_tcp.astype(np.float32)}, "target": target_elbow.astype(np.float32)}
                all_samples.append(sample)
        
        # 计算并保存归一化统计数据
        print("\nCalculating normalization stats...")
        temp_dataset = HandElbowTrajectoryDataset(all_samples)
        all_inputs_np = np.array([item[0].numpy() for item in temp_dataset])
        mean = all_inputs_np.mean(axis=0, dtype=np.float32)
        std = all_inputs_np.std(axis=0, dtype=np.float32)
        std[std < 1e-6] = 1.0
        stats = {"mean": mean, "std": std}
        stats_save_path = save_path.replace('.pt', '_stats.pkl')
        joblib.dump(stats, stats_save_path)
        print(f"📊 Normalization stats saved to {stats_save_path}")
        
        torch.save(all_samples, save_path)
        print(f"\n💾 Saved {len(all_samples)} training samples to {save_path}")

if __name__ == "__main__":
    wandb.init(project="elbow_mlp_tuning", name="exp_with_normalization_fixed")
    hand_elbow_trajectory_store_path = '/home/hjj/human2humanoid/data/new_robot/amass_all.pkl'
    preprocessed_save_path = '/home/hjj/human2humanoid/hand_elbow_training_data_10.pt'
    model_save_path = "/home/hjj/human2humanoid/elbow_mlp_model_10.pth"
    
    run_preprocessing = True
    if run_preprocessing or not os.path.exists(preprocessed_save_path):
        print("--- 运行数据预处理 ---")
        ElbowNNWrapper.preprocess_from_pkl(pkl_path=hand_elbow_trajectory_store_path, save_path=preprocessed_save_path)
    
    print("--- 划分训练集和验证集 ---")
    all_samples = torch.load(preprocessed_save_path)
    random.shuffle(all_samples); val_split = int(len(all_samples) * 0.1)
    val_samples, train_samples = all_samples[:val_split], all_samples[val_split:]
    train_dataset = HandElbowTrajectoryDataset(train_samples)
    val_dataset = HandElbowTrajectoryDataset(val_samples)
    print(f"数据划分完成: {len(train_dataset)} 训练样本, {len(val_dataset)} 验证样本")

    print("\n--- 开始模型训练 ---")
    trainer = ElbowNNWrapper(lr=1e-3)
    trainer.train(train_dataset, val_dataset, epochs=1000, batch_size=256, patience=30)
    
    if os.path.exists("best_model.pth"):
        os.rename("best_model.pth", model_save_path)
        stats_train_path = preprocessed_save_path.replace('.pt', '_stats.pkl')
        stats_model_path = model_save_path.replace('.pth', '_stats.pkl')
        os.system(f'cp "{stats_train_path}" "{stats_model_path}"')
        print(f"✅ Best model and stats saved to {model_save_path} and {stats_model_path}")
    else:
        print("⚠️ No best model was saved.")
    wandb.finish()