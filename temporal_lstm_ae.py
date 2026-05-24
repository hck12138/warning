import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Utilities / 工具函数
# =========================
def robust_threshold(x, q=0.95, mad_k=3.0):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return np.nan

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    qv = float(np.quantile(x, q))

    if mad < 1e-12:
        return qv

    return max(qv, med + mad_k * 1.4826 * mad)


def count_alarm_episodes(alarm_series):
    alarm = alarm_series.astype(bool).values
    count = 0
    prev = False

    for v in alarm:
        if v and not prev:
            count += 1
        prev = v

    return count


def first_alarm_before_event(df, time_col, alarm_col, event_time):
    before = df[(df[time_col] <= event_time) & (df[alarm_col].astype(bool))]
    if len(before) == 0:
        return None, None

    first_time = before[time_col].iloc[0]
    warning_min = (event_time - first_time).total_seconds() / 60.0
    return first_time, warning_min


def get_alarm_metrics(df, time_col, alarm_col, event_time, no_stuck=False):
    alarm_series = df[alarm_col].astype(bool)
    alarm_rows = df[alarm_series]
    first_time = None if len(alarm_rows) == 0 else alarm_rows[time_col].iloc[0]

    if no_stuck:
        # No-stuck well: every alarm episode is counted as non-accident alarm.
        # 无卡钻井：所有报警段都统计为非事故报警。
        return first_time, np.nan, count_alarm_episodes(alarm_series)

    first_before, warning_min = first_alarm_before_event(df, time_col, alarm_col, event_time)

    false_region = df[df[time_col] < (event_time - pd.Timedelta(minutes=60))]
    false_alarm_episodes = count_alarm_episodes(false_region[alarm_col])

    return first_before, warning_min, false_alarm_episodes


def add_persistent_alarm(raw_alarm, persistence_points):
    return (
        raw_alarm.astype(int)
        .rolling(window=persistence_points, min_periods=persistence_points)
        .sum()
        >= persistence_points
    ).fillna(False)


# =========================
# Dataset / 序列数据集
# =========================
class SequenceDataset(Dataset):
    def __init__(self, x_array, end_indices, seq_len):
        self.x = x_array.astype(np.float32)
        self.end_indices = end_indices
        self.seq_len = seq_len

    def __len__(self):
        return len(self.end_indices)

    def __getitem__(self, idx):
        end = self.end_indices[idx]
        start = end - self.seq_len + 1
        seq = self.x[start:end + 1]
        return torch.from_numpy(seq), end


# =========================
# LSTM Autoencoder / LSTM自编码器
# =========================
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, num_layers=1):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        _, (h_n, _) = self.encoder(x)

        h_last = h_n[-1]
        z = self.to_latent(h_last)
        h_dec0 = self.from_latent(z)

        # Repeat hidden state across the sequence length.
        # 将隐藏状态复制成序列，用于重构整个窗口。
        repeated = h_dec0.unsqueeze(1).repeat(1, x.size(1), 1)

        dec_out, _ = self.decoder(repeated)
        x_hat = self.output_layer(dec_out)

        return x_hat


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--residual_csv", type=str, required=True)
    parser.add_argument("--time_col", type=str, default="date")
    parser.add_argument("--event_time", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/temporal_lstm_ae")

    parser.add_argument("--no_stuck", action="store_true")
    parser.add_argument("--train_gap_minutes", type=float, default=60.0)
    parser.add_argument("--train_ratio_no_stuck", type=float, default=0.7)

    parser.add_argument("--sampling_seconds", type=float, default=4.0)
    parser.add_argument("--seq_minutes", type=float, default=15.0)
    parser.add_argument("--stride_points", type=int, default=1)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--threshold_q", type=float, default=0.95)
    parser.add_argument("--mad_k", type=float, default=3.0)
    parser.add_argument("--persistence_points", type=int, default=15)
    parser.add_argument("--temporal_lookback_minutes", type=float, default=15.0)

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.residual_csv)

    if args.time_col not in df.columns:
        raise ValueError(f"Time column not found: {args.time_col}")

    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
    df = df.dropna(subset=[args.time_col]).sort_values(args.time_col).reset_index(drop=True)

    if args.event_time is None:
        event_time = df[args.time_col].max()
        print(f"[INFO] event_time not provided. Using max timestamp: {event_time}")
    else:
        event_time = pd.to_datetime(args.event_time)

    # =========================
    # Feature columns / 输入特征
    # =========================
    candidate_cols = [
        "Average Standpipe Pressure kPa",
        "Mud Flow In L/min",
        "Average Surface Torque kN.m",
        "Average Rotary Speed rpm",
        "Weight on Bit kkgf",
        "Rate of Penetration m/h",
        "Average Hookload kkgf",
        "Bit Depth (MD) m",

        # Mechanism residual scores / 机理残差分数
        "R_hyd_score",
        "R_torque_score",
        "R_eff_score",
        "R_hook_score",
        "R_confirm_score",
        "R_mech_trend_score",
    ]

    feature_cols = [c for c in candidate_cols if c in df.columns]

    if len(feature_cols) < 5:
        raise ValueError(f"Too few available feature columns: {feature_cols}")

    print("=" * 80)
    print("Temporal LSTM-AE input features:")
    for c in feature_cols:
        print(" -", c)
    print("=" * 80)

    # =========================
    # Train period / 训练段
    # =========================
    if args.no_stuck:
        # For no-stuck wells, train on the early part only.
        # 无卡钻井：只用前 train_ratio_no_stuck 作为正常训练段。
        train_end_idx = int(len(df) * args.train_ratio_no_stuck)
        train_mask = df.index < train_end_idx
        train_end_time = df.loc[train_end_idx, args.time_col] if train_end_idx < len(df) else df[args.time_col].max()
    else:
        train_end_time = event_time - pd.Timedelta(minutes=args.train_gap_minutes)
        train_mask = df[args.time_col] < train_end_time

    print(f"Rows total: {len(df)}")
    print(f"Training rows: {int(train_mask.sum())}")
    print(f"Event time: {event_time}")
    print(f"Train end time: {train_end_time}")

    # =========================
    # Clean and normalize / 清洗与归一化
    # =========================
    x_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()

    train_raw = x_raw.loc[train_mask]
    med = train_raw.median()
    iqr = train_raw.quantile(0.75) - train_raw.quantile(0.25)
    iqr = iqr.replace(0, 1.0)

    x_filled = x_raw.fillna(med)
    x_norm = (x_filled - med) / iqr
    x_norm = x_norm.fillna(0.0)

    x_array = x_norm.values.astype(np.float32)

    # =========================
    # Build sequences / 构造滑动窗口序列
    # =========================
    seq_len = max(2, int(args.seq_minutes * 60 / args.sampling_seconds))

    all_end_indices = np.arange(seq_len - 1, len(df), args.stride_points)

    train_end_indices = [
        i for i in all_end_indices
        if train_mask.iloc[i]
    ]

    if len(train_end_indices) < 100:
        raise ValueError(f"Not enough training sequences: {len(train_end_indices)}")

    all_dataset = SequenceDataset(x_array, all_end_indices, seq_len)
    train_dataset = SequenceDataset(x_array, np.array(train_end_indices), seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    all_loader = DataLoader(
        all_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # =========================
    # Train model / 训练模型
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMAutoEncoder(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []

        for seq, _ in train_loader:
            seq = seq.to(device)

            optimizer.zero_grad()
            seq_hat = model(seq)
            loss = loss_fn(seq_hat, seq)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch:03d}/{args.epochs}, train_loss={np.mean(losses):.6f}")

    # =========================
    # Score all windows / 计算全序列时序异常分数
    # =========================
    model.eval()

    score_by_end = {}

    with torch.no_grad():
        for seq, end_idx in all_loader:
            seq = seq.to(device)
            seq_hat = model(seq)

            # Mean absolute reconstruction error over all time steps and variables.
            # 对整个窗口所有时间点和变量计算平均绝对重构误差。
            err = torch.mean(torch.abs(seq_hat - seq), dim=(1, 2)).cpu().numpy()

            for idx, e in zip(end_idx.numpy(), err):
                score_by_end[int(idx)] = float(e)

    df["S_temp_score"] = np.nan
    for idx, score in score_by_end.items():
        df.loc[idx, "S_temp_score"] = score

    # Fill early warm-up region.
    # 前 seq_len-1 个点没有完整窗口，保留 NaN。
    train_scores = df.loc[train_mask, "S_temp_score"].dropna()

    temp_threshold = robust_threshold(
        train_scores,
        q=args.threshold_q,
        mad_k=args.mad_k,
    )

    df["S_temp_alarm_raw"] = df["S_temp_score"] > temp_threshold
    df["S_temp_alarm"] = add_persistent_alarm(
        df["S_temp_alarm_raw"],
        args.persistence_points,
    )

    # =========================
    # Fusion with mechanism yellow / 与机理黄色预警融合
    # =========================
    if "R_confirm_yellow" in df.columns:
        temporal_lookback_points = max(
            1,
            int(args.temporal_lookback_minutes * 60 / args.sampling_seconds)
        )

        temp_recent = (
            df["S_temp_alarm"]
            .astype(int)
            .rolling(temporal_lookback_points, min_periods=1)
            .max()
            .astype(bool)
        )

        df["R_temporal_red_raw"] = (
            df["R_confirm_yellow"].astype(bool)
            & temp_recent
        )

        df["R_temporal_red"] = add_persistent_alarm(
            df["R_temporal_red_raw"],
            args.persistence_points,
        )
    else:
        df["R_temporal_red_raw"] = False
        df["R_temporal_red"] = False

    # =========================
    # Summary / 汇总指标
    # =========================
    summary_rows = []

    for alarm_col, meaning in [
        ("S_temp_alarm", "Temporal anomaly alarm from LSTM-AE reconstruction error"),
        ("R_temporal_red", "Temporal red confirmation: R_confirm_yellow & recent S_temp_alarm"),
    ]:
        first_time, warning_min, false_episodes = get_alarm_metrics(
            df=df,
            time_col=args.time_col,
            alarm_col=alarm_col,
            event_time=event_time,
            no_stuck=args.no_stuck,
        )

        summary_rows.append({
            "metric": alarm_col,
            "meaning": meaning,
            "threshold": temp_threshold if alarm_col == "S_temp_alarm" else "rule_based",
            "first_alarm_time": first_time,
            "warning_min": warning_min,
            "false_alarm_episodes": false_episodes,
            "train_rows": int(train_mask.sum()),
            "eval_rows": len(df),
            "seq_len_points": seq_len,
            "seq_minutes": args.seq_minutes,
            "feature_count": len(feature_cols),
        })

        print(f"[DONE] {alarm_col}: first_alarm={first_time}, warning_min={warning_min}, false_episodes={false_episodes}")

    summary = pd.DataFrame(summary_rows)

    # =========================
    # Save outputs / 保存结果
    # =========================
    output_csv = output_dir / "temporal_outputs.csv"
    summary_csv = output_dir / "temporal_summary.csv"

    df.to_csv(output_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    # Plot temporal score.
    # 画时序异常分数。
    plt.figure(figsize=(12, 4))
    plt.plot(df[args.time_col], df["S_temp_score"], label="S_temp_score")
    plt.axhline(temp_threshold, linestyle="--", label="temporal threshold")

    if not args.no_stuck:
        plt.axvline(event_time, linestyle="--", label="stuck time")

    if "R_temporal_red" in df.columns:
        alarm_area = df["R_temporal_red"].astype(bool)
        plt.fill_between(
            df[args.time_col],
            0,
            df["S_temp_score"].fillna(0),
            where=alarm_area,
            alpha=0.25,
            label="R_temporal_red",
        )

    plt.title("Temporal LSTM-AE anomaly score")
    plt.xlabel("Time")
    plt.ylabel("Reconstruction error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "S_temp_score.png", dpi=200)
    plt.close()

    print("=" * 80)
    print(f"Saved temporal outputs to: {output_csv}")
    print(f"Saved temporal summary to: {summary_csv}")
    print(f"Saved figure to: {output_dir / 'S_temp_score.png'}")
    print("=" * 80)


if __name__ == "__main__":
    main()