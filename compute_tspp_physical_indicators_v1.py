#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute literature-guided stuck-pipe physical indicators for TSPP datasets.
计算适配 TSPP 数据集的卡钻物理指标（参考 Brankovic et al., 2021）。

Supported indicators / 支持的指标:
1) D_lin   : linear-motion issue indicator, based on Block Position + Hookload
             线性运动异常指标，基于游车位置与大钩载荷
2) D_rot   : rotational-motion issue indicator, based on Torque + RPM
             旋转运动异常指标，基于扭矩与转速
3) D_press : adapted pressure-anomaly indicator, based on Standpipe Pressure + Mud Flow In
             压力异常改编指标，基于立管压力与入口流量

Important / 重要说明:
- The original paper used 5 s sampled mudlog data. TSPP uses 4 s data.
  原论文数据采样间隔为 5 s，TSPP 数据为 4 s。
- This script keeps the real 4 s sampling interval and converts paper windows
  by physical time length, instead of pretending that TSPP is 5 s sampled.
  本脚本保留真实的 4 s 采样间隔，并按真实时间长度换算窗口点数。
- D_lin and D_rot follow the physical definitions of the paper.
  D_lin 与 D_rot 依据论文中的物理定义实现。
- D_press is an adapted version because TSPP has Mud Flow In but not SPMT.
  由于 TSPP 没有 SPMT，D_press 使用 Mud Flow In 作为泵工况代理变量，是改编版而非严格复现。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -----------------------------
# Column names / 列名
# -----------------------------
DATE_COL = "date"
HOOKLOAD_COL = "Average Hookload kkgf"
BPOS_COL = "Block Position m"
RPM_COL = "Average Rotary Speed rpm"
TORQUE_COL = "Average Surface Torque kN.m"
SPP_COL = "Average Standpipe Pressure kPa"
FLOW_COL = "Mud Flow In L/min"


REQUIRED_COLUMNS = [
    DATE_COL,
    HOOKLOAD_COL,
    BPOS_COL,
    RPM_COL,
    TORQUE_COL,
    SPP_COL,
    FLOW_COL,
]


# -----------------------------
# Utilities / 工具函数
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute literature-guided stuck-pipe physical indicators for TSPP."
    )
    parser.add_argument("--dataset", type=str, default="TSPP2")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--sample_seconds", type=float, default=4.0)

    parser.add_argument("--normal_start", type=str, required=True)
    parser.add_argument("--normal_end", type=str, required=True)
    parser.add_argument("--stuck_time", type=str, default=None)

    parser.add_argument("--plot_start", type=str, default=None)
    parser.add_argument("--plot_end", type=str, default=None)

    # D_lin settings / D_lin 设置
    # The paper text says "35-second window"; here we follow the time-equivalent interpretation.
    # 论文正文写为 35 秒窗口；这里按等效物理时间换算。
    parser.add_argument("--dlin_median_seconds", type=float, default=35.0)
    parser.add_argument("--dlin_corr_seconds", type=float, default=300.0)  # 5 min
    parser.add_argument("--dlin_corr_threshold", type=float, default=0.25)

    # D_rot settings / D_rot 设置
    parser.add_argument("--drot_window_seconds", type=float, default=100.0)
    parser.add_argument("--drot_persistence_ratio", type=float, default=0.90)
    parser.add_argument(
        "--drot_ratio_threshold",
        type=float,
        default=None,
        help="Fixed threshold for Torque/RPM ratio. If omitted, calibrated from the normal window.",
    )
    parser.add_argument(
        "--drot_ratio_quantile",
        type=float,
        default=0.95,
        help="Normal-window quantile used to calibrate D_rot threshold when fixed threshold is omitted.",
    )
    parser.add_argument("--rpm_floor", type=float, default=1.0)

    # D_press settings / D_press 设置
    parser.add_argument("--dpress_window_seconds", type=float, default=500.0)
    parser.add_argument(
        "--flow_std_threshold",
        type=float,
        default=None,
        help="Fixed threshold for flow stability. If omitted, calibrated from normal-window rolling std.",
    )
    parser.add_argument(
        "--flow_std_quantile",
        type=float,
        default=0.95,
        help="Normal-window quantile used to calibrate flow stability threshold.",
    )
    parser.add_argument("--flow_nonzero_min", type=float, default=10.0)
    parser.add_argument(
        "--dpress_score_threshold",
        type=float,
        default=None,
        help="Fixed threshold for rolling SPP std. If omitted, calibrated from normal window.",
    )
    parser.add_argument(
        "--dpress_score_quantile",
        type=float,
        default=0.95,
        help="Normal-window quantile used to calibrate rolling SPP std threshold.",
    )
    parser.add_argument(
        "--spp_delta_threshold",
        type=float,
        default=None,
        help="Fixed threshold for SPP above rolling median. If omitted, calibrated from normal window.",
    )
    parser.add_argument(
        "--spp_delta_quantile",
        type=float,
        default=0.95,
        help="Normal-window quantile used to calibrate positive SPP delta threshold.",
    )

    # Optional paper-style prefilter / 可选预过滤
    parser.add_argument(
        "--hookload_min",
        type=float,
        default=None,
        help="Optional conservative prefilter for low-hookload stand-change-like samples. "
             "If omitted, no hookload prefilter is applied.",
    )

    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def ensure_odd(n: int) -> int:
    """Ensure centered rolling median window is odd. / 保证居中滑窗为奇数。"""
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def seconds_to_points(seconds: float, sample_seconds: float, *, odd: bool = False) -> int:
    """Convert physical time length to sample points. / 将物理时间换算为采样点数。"""
    points = max(1, int(round(seconds / sample_seconds)))
    return ensure_odd(points) if odd else points


def safe_quantile(values: np.ndarray, q: float, fallback: float = np.nan) -> float:
    """Quantile with finite-value protection. / 带有限值保护的分位数。"""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return fallback
    return float(np.quantile(arr, q))


def detrend_1d(y: np.ndarray) -> np.ndarray:
    """Remove linear trend from a 1D signal. / 去除一维信号的线性趋势。"""
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return y - np.nanmean(y)
    coef = np.polyfit(x[mask], y[mask], 1)
    trend = coef[0] * x + coef[1]
    return y - trend


def rolling_corr_with_detrended_y(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Rolling correlation corr(x, detrend(y)).
    滚动计算 corr(x, detrend(y))。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(len(x), np.nan, dtype=float)

    for end in range(window - 1, len(x)):
        start = end - window + 1
        xw = x[start : end + 1]
        yw = y[start : end + 1]
        mask = np.isfinite(xw) & np.isfinite(yw)
        if mask.sum() < 3:
            continue
        x_use = xw[mask]
        y_use = detrend_1d(yw[mask])
        if np.nanstd(x_use) <= 1e-12 or np.nanstd(y_use) <= 1e-12:
            continue
        out[end] = float(np.corrcoef(x_use, y_use)[0, 1])

    return out


def read_data(path: str) -> pd.DataFrame:
    """Read and validate TSPP CSV. / 读取并检查 TSPP CSV。"""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    numeric_cols = [c for c in REQUIRED_COLUMNS if c != DATE_COL]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def time_mask(df: pd.DataFrame, start: str, end: str) -> np.ndarray:
    """Build time mask [start, end]. / 构造闭区间时间掩码。"""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return ((df[DATE_COL] >= start_ts) & (df[DATE_COL] <= end_ts)).to_numpy()


# -----------------------------
# Indicator computation / 指标计算
# -----------------------------
def compute_dlin(
    df: pd.DataFrame,
    median_points: int,
    corr_points: int,
    corr_threshold: float,
    valid_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    D_lin: linear-motion issue indicator.
    线性运动异常指标：Block Position 与 Hookload 高频分量相关性。
    """
    bpos = df[BPOS_COL].to_numpy(dtype=float)
    hookload = df[HOOKLOAD_COL].to_numpy(dtype=float)

    bpos_median = (
        pd.Series(bpos)
        .rolling(window=median_points, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    hookload_median = (
        pd.Series(hookload)
        .rolling(window=median_points, center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    bpos_diff = bpos - bpos_median
    hookload_diff = hookload - hookload_median

    dlin_corr = rolling_corr_with_detrended_y(
        x=bpos_diff,
        y=hookload_diff,
        window=corr_points,
    )
    dlin_flag = dlin_corr >= corr_threshold

    if valid_mask is not None:
        dlin_flag = dlin_flag & valid_mask

    out = pd.DataFrame(
        {
            "dlin_bpos_median": bpos_median,
            "dlin_hookload_median": hookload_median,
            "dlin_bpos_diff": bpos_diff,
            "dlin_hookload_diff": hookload_diff,
            "dlin_score_corr": dlin_corr,
            "dlin_flag": dlin_flag.astype(int),
        }
    )
    return out


def compute_drot(
    df: pd.DataFrame,
    normal_mask: np.ndarray,
    window_points: int,
    persistence_ratio: float,
    ratio_threshold: Optional[float],
    ratio_quantile: float,
    rpm_floor: float,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    D_rot: rotational-motion issue indicator.
    旋转运动异常指标：高 Torque / 低 RPM。
    """
    torque = np.abs(df[TORQUE_COL].to_numpy(dtype=float))
    rpm = np.abs(df[RPM_COL].to_numpy(dtype=float))
    drot_ratio = torque / np.maximum(rpm, rpm_floor)

    if ratio_threshold is None:
        ratio_threshold = safe_quantile(drot_ratio[normal_mask], ratio_quantile)
    if not np.isfinite(ratio_threshold):
        raise ValueError("Unable to calibrate D_rot threshold from the normal window.")

    ratio_high = drot_ratio >= ratio_threshold
    required_points = int(np.ceil(window_points * persistence_ratio))
    drot_count = (
        pd.Series(ratio_high.astype(int))
        .rolling(window=window_points, min_periods=window_points)
        .sum()
        .to_numpy()
    )
    drot_flag = drot_count >= required_points

    if valid_mask is not None:
        drot_flag = drot_flag & valid_mask

    out = pd.DataFrame(
        {
            "drot_ratio": drot_ratio,
            "drot_ratio_high": ratio_high.astype(int),
            "drot_count": drot_count,
            "drot_flag": drot_flag.astype(int),
        }
    )
    meta = {
        "drot_ratio_threshold": float(ratio_threshold),
        "drot_window_points": int(window_points),
        "drot_required_points": int(required_points),
    }
    return out, meta


def compute_dpress_flow(
    df: pd.DataFrame,
    normal_mask: np.ndarray,
    window_points: int,
    flow_std_threshold: Optional[float],
    flow_std_quantile: float,
    flow_nonzero_min: float,
    score_threshold: Optional[float],
    score_quantile: float,
    spp_delta_threshold: Optional[float],
    spp_delta_quantile: float,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Adapted D_press indicator using Mud Flow In as a proxy for pump condition.
    使用入口流量代替 SPMT 的改编版压力异常指标。
    """
    flow = df[FLOW_COL].to_numpy(dtype=float)
    spp = df[SPP_COL].to_numpy(dtype=float)

    flow_series = pd.Series(flow)
    spp_series = pd.Series(spp)

    # Trailing rolling window / 采用向后滑窗，便于后续在线化
    flow_mean = flow_series.rolling(window=window_points, min_periods=window_points).mean().to_numpy()
    flow_std = flow_series.rolling(window=window_points, min_periods=window_points).std(ddof=0).to_numpy()

    # Median baseline for SPP / SPP 的滑动中位数基线
    spp_median = spp_series.rolling(window=window_points, center=True, min_periods=1).median().to_numpy()
    spp_delta = spp - spp_median
    spp_std = spp_series.rolling(window=window_points, min_periods=window_points).std(ddof=0).to_numpy()

    if flow_std_threshold is None:
        flow_std_threshold = safe_quantile(flow_std[normal_mask], flow_std_quantile)
    if not np.isfinite(flow_std_threshold):
        raise ValueError("Unable to calibrate flow stability threshold from the normal window.")

    flow_stable = (flow_std <= flow_std_threshold) & (flow_mean > flow_nonzero_min)

    dpress_score = np.where(flow_stable, spp_std, 0.0)

    if score_threshold is None:
        candidate = dpress_score[normal_mask & flow_stable]
        score_threshold = safe_quantile(candidate, score_quantile)
    if not np.isfinite(score_threshold):
        raise ValueError("Unable to calibrate D_press score threshold from the normal window.")

    if spp_delta_threshold is None:
        # Use positive deltas only if available / 优先用正向增量
        positive_delta = spp_delta[normal_mask & np.isfinite(spp_delta) & (spp_delta > 0)]
        if positive_delta.size > 0:
            spp_delta_threshold = safe_quantile(positive_delta, spp_delta_quantile)
        else:
            spp_delta_threshold = safe_quantile(spp_delta[normal_mask], spp_delta_quantile, fallback=0.0)
    if not np.isfinite(spp_delta_threshold):
        raise ValueError("Unable to calibrate SPP delta threshold from the normal window.")

    dpress_flag = (
        flow_stable
        & (dpress_score >= score_threshold)
        & (spp_delta >= spp_delta_threshold)
    )

    if valid_mask is not None:
        dpress_flag = dpress_flag & valid_mask

    out = pd.DataFrame(
        {
            "dpress_flow_mean": flow_mean,
            "dpress_flow_std": flow_std,
            "dpress_flow_stable": flow_stable.astype(int),
            "dpress_spp_median": spp_median,
            "dpress_spp_delta": spp_delta,
            "dpress_score_spp_std": dpress_score,
            "dpress_flag": dpress_flag.astype(int),
        }
    )
    meta = {
        "dpress_flow_std_threshold": float(flow_std_threshold),
        "dpress_score_threshold": float(score_threshold),
        "dpress_spp_delta_threshold": float(spp_delta_threshold),
        "dpress_window_points": int(window_points),
    }
    return out, meta


# -----------------------------
# Plotting / 绘图
# -----------------------------
def prepare_plot_df(df: pd.DataFrame, plot_start: Optional[str], plot_end: Optional[str]) -> pd.DataFrame:
    out = df
    if plot_start is not None:
        out = out[out[DATE_COL] >= pd.Timestamp(plot_start)]
    if plot_end is not None:
        out = out[out[DATE_COL] <= pd.Timestamp(plot_end)]
    return out.copy()


def add_time_format(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.tick_params(axis="x", rotation=25)


def add_stuck_line(ax: plt.Axes, stuck_time: Optional[str]) -> None:
    if stuck_time is not None:
        ax.axvline(pd.Timestamp(stuck_time), linestyle="--", linewidth=1.2, label="Stuck time")


def plot_overview(
    df: pd.DataFrame,
    out_path: Path,
    dataset: str,
    dlin_threshold: float,
    drot_threshold: float,
    dpress_threshold: float,
    stuck_time: Optional[str],
    dpi: int,
) -> None:
    p = df
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)

    axes[0].plot(p[DATE_COL], p["dlin_score_corr"], label="D_lin corr score")
    axes[0].axhline(dlin_threshold, linestyle="--", label="D_lin threshold")
    axes[0].fill_between(
        p[DATE_COL],
        0,
        p["dlin_flag"].to_numpy(dtype=float),
        alpha=0.15,
        label="D_lin flag",
    )
    axes[0].set_ylabel("D_lin")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)
    add_stuck_line(axes[0], stuck_time)

    axes[1].plot(p[DATE_COL], p["drot_ratio"], label="D_rot ratio")
    axes[1].axhline(drot_threshold, linestyle="--", label="D_rot threshold")
    drot_peak = np.nanmax(p["drot_ratio"].to_numpy(dtype=float))
    if not np.isfinite(drot_peak) or drot_peak <= 0:
        drot_peak = 1.0
    axes[1].fill_between(
        p[DATE_COL],
        0,
        p["drot_flag"].to_numpy(dtype=float) * drot_peak,
        alpha=0.12,
        label="D_rot flag",
    )
    axes[1].set_ylabel("D_rot ratio")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)
    add_stuck_line(axes[1], stuck_time)

    axes[2].plot(p[DATE_COL], p["dpress_score_spp_std"], label="D_press_flow score")
    axes[2].axhline(dpress_threshold, linestyle="--", label="D_press_flow threshold")
    dpress_peak = np.nanmax(p["dpress_score_spp_std"].to_numpy(dtype=float))
    if not np.isfinite(dpress_peak) or dpress_peak <= 0:
        dpress_peak = 1.0
    axes[2].fill_between(
        p[DATE_COL],
        0,
        p["dpress_flag"].to_numpy(dtype=float) * dpress_peak,
        alpha=0.12,
        label="D_press_flow flag",
    )
    axes[2].set_ylabel("D_press_flow")
    axes[2].set_xlabel("Time")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.25)
    add_stuck_line(axes[2], stuck_time)

    for ax in axes:
        add_time_format(ax)

    fig.suptitle(f"{dataset} literature-guided physical indicators")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_dlin_detail(
    df: pd.DataFrame,
    out_path: Path,
    dataset: str,
    corr_threshold: float,
    stuck_time: Optional[str],
    dpi: int,
) -> None:
    p = df
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(p[DATE_COL], p[BPOS_COL], label=BPOS_COL)
    axes[0].set_ylabel("BPOS")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)
    add_stuck_line(axes[0], stuck_time)

    axes[1].plot(p[DATE_COL], p[HOOKLOAD_COL], label=HOOKLOAD_COL)
    axes[1].set_ylabel("Hookload")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)
    add_stuck_line(axes[1], stuck_time)

    axes[2].plot(p[DATE_COL], p["dlin_score_corr"], label="D_lin corr score")
    axes[2].axhline(corr_threshold, linestyle="--", label="Threshold")
    axes[2].set_ylabel("Correlation")
    axes[2].set_xlabel("Time")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.25)
    add_stuck_line(axes[2], stuck_time)

    for ax in axes:
        add_time_format(ax)
    fig.suptitle(f"{dataset} D_lin detail")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_drot_detail(
    df: pd.DataFrame,
    out_path: Path,
    dataset: str,
    ratio_threshold: float,
    stuck_time: Optional[str],
    dpi: int,
) -> None:
    p = df
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(p[DATE_COL], p[TORQUE_COL], label=TORQUE_COL)
    axes[0].set_ylabel("Torque")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)
    add_stuck_line(axes[0], stuck_time)

    axes[1].plot(p[DATE_COL], p[RPM_COL], label=RPM_COL)
    axes[1].set_ylabel("RPM")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)
    add_stuck_line(axes[1], stuck_time)

    axes[2].plot(p[DATE_COL], p["drot_ratio"], label="Torque / max(RPM, floor)")
    axes[2].axhline(ratio_threshold, linestyle="--", label="Threshold")
    axes[2].set_ylabel("Ratio")
    axes[2].set_xlabel("Time")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.25)
    add_stuck_line(axes[2], stuck_time)

    for ax in axes:
        add_time_format(ax)
    fig.suptitle(f"{dataset} D_rot detail")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_dpress_detail(
    df: pd.DataFrame,
    out_path: Path,
    dataset: str,
    score_threshold: float,
    stuck_time: Optional[str],
    dpi: int,
) -> None:
    p = df
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    axes[0].plot(p[DATE_COL], p[FLOW_COL], label=FLOW_COL)
    axes[0].set_ylabel("Flow In")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)
    add_stuck_line(axes[0], stuck_time)

    axes[1].plot(p[DATE_COL], p[SPP_COL], label=SPP_COL)
    axes[1].plot(p[DATE_COL], p["dpress_spp_median"], label="SPP rolling median")
    axes[1].set_ylabel("SPP")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)
    add_stuck_line(axes[1], stuck_time)

    axes[2].plot(p[DATE_COL], p["dpress_flow_std"], label="Flow rolling std")
    axes[2].set_ylabel("Flow std")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.25)
    add_stuck_line(axes[2], stuck_time)

    axes[3].plot(p[DATE_COL], p["dpress_score_spp_std"], label="D_press_flow score")
    axes[3].axhline(score_threshold, linestyle="--", label="Threshold")
    axes[3].set_ylabel("SPP std")
    axes[3].set_xlabel("Time")
    axes[3].legend(loc="upper left")
    axes[3].grid(alpha=0.25)
    add_stuck_line(axes[3], stuck_time)

    for ax in axes:
        add_time_format(ax)
    fig.suptitle(f"{dataset} D_press_flow detail")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main / 主流程
# -----------------------------
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_data(args.data_path)
    normal_mask = time_mask(df, args.normal_start, args.normal_end)

    if normal_mask.sum() == 0:
        raise ValueError("Normal window is empty. Please check --normal_start and --normal_end.")

    valid_mask = None
    if args.hookload_min is not None:
        valid_mask = (df[HOOKLOAD_COL].to_numpy(dtype=float) >= args.hookload_min)

    # Convert paper time scales to 4-s points / 将论文时间尺度换算为 4 秒采样点数
    dlin_median_points = seconds_to_points(args.dlin_median_seconds, args.sample_seconds, odd=True)
    dlin_corr_points = seconds_to_points(args.dlin_corr_seconds, args.sample_seconds, odd=False)
    drot_window_points = seconds_to_points(args.drot_window_seconds, args.sample_seconds, odd=False)
    dpress_window_points = seconds_to_points(args.dpress_window_seconds, args.sample_seconds, odd=False)

    dlin_df = compute_dlin(
        df=df,
        median_points=dlin_median_points,
        corr_points=dlin_corr_points,
        corr_threshold=args.dlin_corr_threshold,
        valid_mask=valid_mask,
    )

    drot_df, drot_meta = compute_drot(
        df=df,
        normal_mask=normal_mask,
        window_points=drot_window_points,
        persistence_ratio=args.drot_persistence_ratio,
        ratio_threshold=args.drot_ratio_threshold,
        ratio_quantile=args.drot_ratio_quantile,
        rpm_floor=args.rpm_floor,
        valid_mask=valid_mask,
    )

    dpress_df, dpress_meta = compute_dpress_flow(
        df=df,
        normal_mask=normal_mask,
        window_points=dpress_window_points,
        flow_std_threshold=args.flow_std_threshold,
        flow_std_quantile=args.flow_std_quantile,
        flow_nonzero_min=args.flow_nonzero_min,
        score_threshold=args.dpress_score_threshold,
        score_quantile=args.dpress_score_quantile,
        spp_delta_threshold=args.spp_delta_threshold,
        spp_delta_quantile=args.spp_delta_quantile,
        valid_mask=valid_mask,
    )

    result = pd.concat([df, dlin_df, drot_df, dpress_df], axis=1)

    # Save all indicator values / 保存所有指标结果
    indicator_csv = out_dir / "physical_indicators.csv"
    result.to_csv(indicator_csv, index=False)

    # Save metadata / 保存参数配置与阈值
    summary = {
        "dataset": args.dataset,
        "data_path": args.data_path,
        "sample_seconds": args.sample_seconds,
        "normal_start": args.normal_start,
        "normal_end": args.normal_end,
        "stuck_time": args.stuck_time,
        "dlin_median_seconds": args.dlin_median_seconds,
        "dlin_median_points": dlin_median_points,
        "dlin_corr_seconds": args.dlin_corr_seconds,
        "dlin_corr_points": dlin_corr_points,
        "dlin_corr_threshold": args.dlin_corr_threshold,
        "drot_window_seconds": args.drot_window_seconds,
        "drot_window_points": drot_window_points,
        "drot_persistence_ratio": args.drot_persistence_ratio,
        "drot_ratio_quantile": args.drot_ratio_quantile,
        **drot_meta,
        "dpress_window_seconds": args.dpress_window_seconds,
        "dpress_window_points": dpress_window_points,
        "flow_std_quantile": args.flow_std_quantile,
        "flow_nonzero_min": args.flow_nonzero_min,
        "dpress_score_quantile": args.dpress_score_quantile,
        "spp_delta_quantile": args.spp_delta_quantile,
        **dpress_meta,
        "hookload_min": args.hookload_min,
        "dlin_positive_count": int(result["dlin_flag"].sum()),
        "drot_positive_count": int(result["drot_flag"].sum()),
        "dpress_positive_count": int(result["dpress_flag"].sum()),
    }
    with open(out_dir / "physical_indicator_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Plot / 绘图
    plot_df = prepare_plot_df(result, args.plot_start, args.plot_end)
    if plot_df.empty:
        raise ValueError("Plot window is empty. Please check --plot_start and --plot_end.")

    plot_overview(
        df=plot_df,
        out_path=out_dir / "physical_indicators_overview.png",
        dataset=args.dataset,
        dlin_threshold=args.dlin_corr_threshold,
        drot_threshold=drot_meta["drot_ratio_threshold"],
        dpress_threshold=dpress_meta["dpress_score_threshold"],
        stuck_time=args.stuck_time,
        dpi=args.dpi,
    )
    plot_dlin_detail(
        df=plot_df,
        out_path=out_dir / "dlin_detail.png",
        dataset=args.dataset,
        corr_threshold=args.dlin_corr_threshold,
        stuck_time=args.stuck_time,
        dpi=args.dpi,
    )
    plot_drot_detail(
        df=plot_df,
        out_path=out_dir / "drot_detail.png",
        dataset=args.dataset,
        ratio_threshold=drot_meta["drot_ratio_threshold"],
        stuck_time=args.stuck_time,
        dpi=args.dpi,
    )
    plot_dpress_detail(
        df=plot_df,
        out_path=out_dir / "dpress_flow_detail.png",
        dataset=args.dataset,
        score_threshold=dpress_meta["dpress_score_threshold"],
        stuck_time=args.stuck_time,
        dpi=args.dpi,
    )

    # Print concise summary / 打印简要结果
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Physical indicators saved to: {indicator_csv}")
    print(f"D_lin  : median_points={dlin_median_points}, corr_points={dlin_corr_points}, "
          f"threshold={args.dlin_corr_threshold}")
    print(f"D_rot  : window_points={drot_window_points}, "
          f"ratio_threshold={drot_meta['drot_ratio_threshold']:.6g}, "
          f"required_points={drot_meta['drot_required_points']}")
    print(f"D_press_flow: window_points={dpress_window_points}, "
          f"flow_std_threshold={dpress_meta['dpress_flow_std_threshold']:.6g}, "
          f"score_threshold={dpress_meta['dpress_score_threshold']:.6g}, "
          f"spp_delta_threshold={dpress_meta['dpress_spp_delta_threshold']:.6g}")
    print(f"Positive counts: D_lin={summary['dlin_positive_count']}, "
          f"D_rot={summary['drot_positive_count']}, "
          f"D_press_flow={summary['dpress_positive_count']}")
    print(f"Figures saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
