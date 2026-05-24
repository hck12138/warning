import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# =========================
# Column names / 列名配置
# =========================
COLS = {
    "spp": "Average Standpipe Pressure kPa",
    "flow": "Mud Flow In L/min",
    "rop": "Rate of Penetration m/h",
    "wob": "Weight on Bit kkgf",
    "rpm": "Average Rotary Speed rpm",
    "torque": "Average Surface Torque kN.m",
    "hook": "Average Hookload kkgf",
    "bit_depth": "Bit Depth (MD) m",
    "block_velocity": "Block Velocity m/s",
    "block_position": "Block Position m",
    "on_bottom": "On Bottom Status unitless",
    "bit_on_bottom": "Bit on Bottom unitless",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 mechanism residual validation for stuck pipe early warning."
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Input TSPP CSV path.")
    parser.add_argument("--time_col", type=str, default="data", help="Time column name. Default: data")
    parser.add_argument("--event_time", type=str, required=True, help="Stuck pipe event time.")
    parser.add_argument("--output_dir", type=str, default="results/mechanism_residual_stage1")

    parser.add_argument(
        "--train_gap_minutes",
        type=float,
        default=60.0,
        help="Do not use data within this many minutes before stuck time for training.",
    )
    parser.add_argument(
        "--plot_minutes_before",
        type=float,
        default=180.0,
        help="Plot this many minutes before stuck time.",
    )
    parser.add_argument(
        "--plot_minutes_after",
        type=float,
        default=10.0,
        help="Plot this many minutes after stuck time.",
    )
    parser.add_argument(
        "--threshold_q",
        type=float,
        default=0.95,
        help="Normal residual quantile threshold.",
    )
    parser.add_argument(
        "--mad_k",
        type=float,
        default=3.0,
        help="MAD multiplier for robust threshold.",
    )
    parser.add_argument(
        "--persistence_points",
        type=int,
        default=5,
        help="Consecutive alarm points required.",
    )
    parser.add_argument(
        "--filter_mode",
        type=str,
        default="loose",
        choices=["loose", "strict"],
        help="loose: on-bottom + flow>0; strict: additionally rpm>10.",
    )


    parser.add_argument(
        "--no_stuck",
        action="store_true",
        help="Use this flag for wells without stuck-pipe events. 用于无卡钻井，不计算 warning_min，只统计误报。"
    )

    parser.add_argument(
        "--confirm_lookback_minutes",
        type=float,
        default=15.0,
        help="Lookback window for mechanism confirmation. 多机理确认的回看窗口。"
    )

    parser.add_argument(
        "--torque_confirm_ratio",
        type=float,
        default=0.6,
        help="Moderate torque threshold ratio for confirmation. 扭矩中等确认阈值比例。"
    )

    parser.add_argument(
        "--rop_low_quantile",
        type=float,
        default=0.2,
        help="ROP low threshold quantile from training data. ROP低值阈值分位数。"
    )

    
    return parser.parse_args()


def robust_threshold(x, q=0.95, mad_k=3.0):
    """
    Robust threshold from normal training residuals.
    使用正常训练段残差计算鲁棒阈值，避免用全段数据造成未来信息泄漏。
    """
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return np.nan

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    qv = float(np.quantile(x, q))

    if mad < 1e-12:
        return qv

    mad_threshold = med + mad_k * 1.4826 * mad
    return max(mad_threshold, qv)


def make_model():
    """
    Random forest normal-response model.
    随机森林正常响应模型：第一版用于验证残差信号，不追求最优模型。
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def check_columns(df, required_cols, residual_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[SKIP] {residual_name}: missing columns: {missing}")
        return False
    return True


def fit_predict_residual(df_eval, df_train, target_col, feature_cols, residual_type):
    """
    Train on normal period and predict residuals on evaluation period.
    只用正常段训练，在评估段计算响应残差。
    """
    model = make_model()

    train_data = df_train[[target_col] + feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_data) < 100:
        raise ValueError(f"Not enough training rows for target {target_col}: {len(train_data)} rows")

    X_train = train_data[feature_cols]
    y_train = train_data[target_col]

    model.fit(X_train, y_train)

    X_eval = df_eval[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_eval = df_eval[target_col].astype(float)

    y_hat = model.predict(X_eval)
    raw_residual = y_eval.values - y_hat

    if residual_type == "positive":
        score = np.maximum(0.0, raw_residual)
    elif residual_type == "absolute":
        score = np.abs(raw_residual)
    elif residual_type == "inverse_positive":
        # For ROP efficiency:
        # predicted ROP - actual ROP > 0 means actual ROP is lower than expected.
        # 对 ROP 效率残差：预测 ROP 高于真实 ROP，说明钻进效率退化。
        score = np.maximum(0.0, -raw_residual)
    else:
        raise ValueError(f"Unknown residual_type: {residual_type}")

    return y_hat, raw_residual, score


def add_persistent_alarm(df, score_col, threshold, persistence_points):
    alarm_raw = df[score_col] > threshold
    alarm_persistent = (
        alarm_raw.astype(int)
        .rolling(window=persistence_points, min_periods=persistence_points)
        .sum()
        >= persistence_points
    )
    return alarm_raw, alarm_persistent.fillna(False)


def first_alarm_before_event(df, time_col, alarm_col, event_time):
    before = df[(df[time_col] <= event_time) & (df[alarm_col])]
    if len(before) == 0:
        return None, None

    first_time = before[time_col].iloc[0]
    warning_min = (event_time - first_time).total_seconds() / 60.0
    return first_time, warning_min


def count_alarm_episodes(alarm_series):
    alarm = alarm_series.astype(bool).values
    if len(alarm) == 0:
        return 0
    starts = 0
    prev = False
    for v in alarm:
        if v and not prev:
            starts += 1
        prev = v
    return starts


def get_alarm_metrics(df, time_col, alarm_col, event_time, no_stuck=False):
    """
    Compute alarm metrics.
    计算报警指标：
    - 有卡钻井：first alarm, warning time, false alarms before 60 min.
    - 无卡钻井：first alarm, warning_min=NaN, all alarm episodes as false alarms.
    """
    alarm_series = df[alarm_col].astype(bool)

    alarm_rows = df[alarm_series]
    first_time = None if len(alarm_rows) == 0 else alarm_rows[time_col].iloc[0]

    if no_stuck:
        # For no-stuck wells, every alarm is a false alarm.
        # 无卡钻井中，所有报警都算误报。
        false_alarm_episodes = count_alarm_episodes(alarm_series)
        return first_time, np.nan, false_alarm_episodes

    first_before, warning_min = first_alarm_before_event(
        df, time_col, alarm_col, event_time
    )

    false_region = df[df[time_col] < (event_time - pd.Timedelta(minutes=60))]
    false_alarm_episodes = count_alarm_episodes(false_region[alarm_col])

    return first_before, warning_min, false_alarm_episodes

def plot_score(df_plot, time_col, score_col, threshold, event_time, output_path, title):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_plot[time_col], df_plot[score_col], label=score_col)
    ax.axhline(threshold, linestyle="--", label="threshold")
    ax.axvline(event_time, linestyle="--", label="stuck time")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Residual score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)



def add_driller_adjust_mask(
    df,
    time_col,
    rpm_col,
    wob_col,
    flow_col=None,
    sampling_seconds=4,
    window_minutes=1.0,
    hold_minutes=3.0,
    rpm_range_threshold=20.0,
    wob_range_threshold=3.0,
    flow_range_threshold=200.0,
):
    """
    Detect active driller parameter adjustments.
    检测司钻主动调参段：短时间内 RPM/WOB/Flow 剧烈变化。
    """

    window_points = max(2, int(window_minutes * 60 / sampling_seconds))
    hold_points = max(1, int(hold_minutes * 60 / sampling_seconds))

    rpm_range = (
        df[rpm_col]
        .rolling(window_points, min_periods=1)
        .max()
        - df[rpm_col].rolling(window_points, min_periods=1).min()
    )

    wob_range = (
        df[wob_col]
        .rolling(window_points, min_periods=1)
        .max()
        - df[wob_col].rolling(window_points, min_periods=1).min()
    )

    adjust_raw = (rpm_range > rpm_range_threshold) | (wob_range > wob_range_threshold)

    if flow_col is not None and flow_col in df.columns:
        flow_range = (
            df[flow_col]
            .rolling(window_points, min_periods=1)
            .max()
            - df[flow_col].rolling(window_points, min_periods=1).min()
        )
        adjust_raw = adjust_raw | (flow_range > flow_range_threshold)

    # Hold mask after adjustment, because the system needs time to settle.
    # 调参后延迟几分钟恢复报警，避免把过渡过程当异常。
    adjust_mask = (
        adjust_raw.astype(int)
        .rolling(hold_points, min_periods=1)
        .max()
        .astype(bool)
    )

    df["driller_adjust_raw"] = adjust_raw
    df["driller_adjust_mask"] = adjust_mask

    return df

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)

    if args.time_col not in df.columns:
        if args.time_col == "data" and "date" in df.columns:
            print("[INFO] time_col 'data' not found, using 'date' instead.")
            args.time_col = "date"
        else:
            raise ValueError(f"Time column '{args.time_col}' not found in CSV.")

    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
    df = df.dropna(subset=[args.time_col]).sort_values(args.time_col).reset_index(drop=True)
    # Fix duplicated minute-level timestamps using 4-second sampling interval
    # 如果时间列只有分钟精度，则用每分钟内的行序号恢复近似 4 秒采样时间
    dup_i = df.groupby(args.time_col).cumcount()
    df[args.time_col] = df[args.time_col] + pd.to_timedelta(dup_i * 4, unit="s")
    # Compute block velocity if missing
    # 如果没有 Block Velocity m/s，就用 Block Position m 根据时间差分计算
    if COLS["block_velocity"] not in df.columns and COLS["block_position"] in df.columns:
        dt = df[args.time_col].diff().dt.total_seconds()
        dt = dt.replace(0, np.nan)

        # Your sampling interval is about 4 seconds
        # 如果时间差分缺失，就用 4 秒采样间隔补充
        dt = dt.fillna(4.0)

        df[COLS["block_velocity"]] = df[COLS["block_position"]].diff() / dt
        df[COLS["block_velocity"]] = df[COLS["block_velocity"]].fillna(0.0)

        print("[INFO] Block Velocity m/s not found. Computed it from Block Position m and time interval.")

    event_time = pd.to_datetime(args.event_time)
    train_end_time = event_time - pd.Timedelta(minutes=args.train_gap_minutes)

    # =========================
    # Rig-state / 工况筛选
    # =========================
    mask = pd.Series(True, index=df.index)

    if COLS["on_bottom"] in df.columns:
        mask &= df[COLS["on_bottom"]] == 1

    if COLS["bit_on_bottom"] in df.columns:
        mask &= df[COLS["bit_on_bottom"]] == 1

    if COLS["flow"] in df.columns:
        mask &= df[COLS["flow"]] > 0

    if args.filter_mode == "strict" and COLS["rpm"] in df.columns:
        mask &= df[COLS["rpm"]] > 10

    # df_eval = df[mask].copy().reset_index(drop=True)
    # df_train = df_eval[df_eval[args.time_col] < train_end_time].copy().reset_index(drop=True)
    df_eval = df[mask].copy().reset_index(drop=True)

    df_eval = add_driller_adjust_mask(
        df=df_eval,
        time_col=args.time_col,
        rpm_col=COLS["rpm"],
        wob_col=COLS["wob"],
        flow_col=COLS["flow"],
        sampling_seconds=4,
        window_minutes=1.0,
        hold_minutes=3.0,
        rpm_range_threshold=20.0,
        wob_range_threshold=3.0,
        flow_range_threshold=200.0,
    )

    # df_train = df_eval[
    #     (df_eval[args.time_col] < train_end_time)
    #     & (~df_eval["driller_adjust_mask"])
    # ].copy().reset_index(drop=True)
    df_train = df_eval[df_eval[args.time_col] < train_end_time].copy().reset_index(drop=True)

    print("=" * 80)
    print(f"Input file: {args.csv_path}")
    print(f"Rows total: {len(df)}")
    print(f"Rows after rig-state filter: {len(df_eval)}")
    print(f"Training rows before {train_end_time}: {len(df_train)}")
    print(f"Event time: {event_time}")
    print("=" * 80)

    residual_configs = {
        "R_hyd": {
            "target": COLS["spp"],
            "features": [COLS["flow"], COLS["rop"], COLS["wob"], COLS["rpm"], COLS["bit_depth"]],
            "type": "positive",
            "meaning": "Hydraulic response residual / 水力响应残差",
        },
        "R_torque": {
            "target": COLS["torque"],
            "features": [COLS["rpm"], COLS["wob"], COLS["rop"], COLS["flow"], COLS["bit_depth"]],
            "type": "positive",
            "meaning": "Torque response residual / 扭矩响应残差",
        },
        "R_hook": {
            "target": COLS["hook"],
            "features": [
                COLS["wob"],
                COLS["block_velocity"],
                COLS["block_position"],
                COLS["bit_depth"],
                COLS["rpm"],
                COLS["flow"],
            ],
            "type": "absolute",
            "meaning": "Hookload response residual / 大钩载响应残差",
        },
        "R_eff": {
            "target": COLS["rop"],
            "features": [COLS["wob"], COLS["rpm"], COLS["torque"], COLS["spp"], COLS["flow"], COLS["bit_depth"]],
            "type": "inverse_positive",
            "meaning": "ROP efficiency residual / 钻进效率退化残差",
        },
    }

    summary_rows = []
    available_scores = []
    threshold_map = {}

    for name, cfg in residual_configs.items():
        required = [cfg["target"]] + cfg["features"]
        if not check_columns(df_eval, required, name):
            continue

        try:
            y_hat, raw_residual, score = fit_predict_residual(
                df_eval=df_eval,
                df_train=df_train,
                target_col=cfg["target"],
                feature_cols=cfg["features"],
                residual_type=cfg["type"],
            )
        except Exception as e:
            print(f"[SKIP] {name}: {e}")
            continue

        pred_col = f"{name}_pred"
        raw_col = f"{name}_raw"
        score_col = f"{name}_score"
        alarm_raw_col = f"{name}_alarm_raw"
        alarm_col = f"{name}_alarm"

        df_eval[pred_col] = y_hat
        df_eval[raw_col] = raw_residual
        df_eval[score_col] = score

        train_scores = df_eval.loc[df_eval[args.time_col] < train_end_time, score_col]
        threshold = robust_threshold(train_scores, q=args.threshold_q, mad_k=args.mad_k)
        threshold_map[name] = float(threshold)

        # df_eval[alarm_raw_col], df_eval[alarm_col] = add_persistent_alarm(
        #     df_eval, score_col, threshold, args.persistence_points
        # )
        df_eval[alarm_raw_col], df_eval[alarm_col] = add_persistent_alarm(
            df_eval, score_col, threshold, args.persistence_points
        )

        # Suppress alarms during active driller adjustments.
        # 司钻主动调参和调参后稳定期内不触发报警。
        # if "driller_adjust_mask" in df_eval.columns:
        #     df_eval[alarm_raw_col] = df_eval[alarm_raw_col] & (~df_eval["driller_adjust_mask"])
        #     df_eval[alarm_col] = df_eval[alarm_col] & (~df_eval["driller_adjust_mask"])


        # first_time, warning_min = first_alarm_before_event(
        #     df_eval, args.time_col, alarm_col, event_time
        # )

        # false_region = df_eval[df_eval[args.time_col] < (event_time - pd.Timedelta(minutes=60))]
        # false_alarm_episodes = count_alarm_episodes(false_region[alarm_col])
        first_time, warning_min, false_alarm_episodes = get_alarm_metrics(
        df=df_eval,
        time_col=args.time_col,
        alarm_col=alarm_col,    
        event_time=event_time,
        no_stuck=args.no_stuck,
    )

        summary_rows.append(
            {
                "residual": name,
                "meaning": cfg["meaning"],
                "threshold": threshold,
                "first_alarm_time": first_time,
                "warning_min": warning_min,
                "false_alarm_episodes_before_60min": false_alarm_episodes,
                "train_rows": len(df_train),
                "eval_rows": len(df_eval),
            }
        )

        available_scores.append((name, score_col, threshold))

        plot_start = event_time - pd.Timedelta(minutes=args.plot_minutes_before)
        plot_end = event_time + pd.Timedelta(minutes=args.plot_minutes_after)
        df_plot = df_eval[
            (df_eval[args.time_col] >= plot_start) & (df_eval[args.time_col] <= plot_end)
        ]

        plot_score(
            df_plot=df_plot,
            time_col=args.time_col,
            score_col=score_col,
            threshold=threshold,
            event_time=event_time,
            output_path=output_dir / f"{name}_score.png",
            title=f"{name}: {cfg['meaning']}",
        )

        print(f"[DONE] {name}: first_alarm={first_time}, warning_min={warning_min}")




    # Smoothed torque alarm for confirmation
    # 用平滑扭矩残差做确认，避免单点波动导致无法连续报警
    # =========================
    # Simple fusion / 简单融合
    # =========================
    # =========================
    # Mechanism-consistency confirmation / 多机理一致性确认
    # =========================
    needed_alarm_cols = [
        "R_hyd_alarm_raw",
        "R_torque_alarm_raw",
        "R_eff_alarm_raw",
    ]

    needed_score_cols = [
        "R_hyd_score",
        "R_torque_score",
        "R_eff_score",
    ]

    if all(c in df_eval.columns for c in needed_alarm_cols + needed_score_cols):
        # Normalize residual scores using their own thresholds
        # 用各自阈值归一化残差，便于画图和比较
        for residual_name in ["R_hyd", "R_torque", "R_eff"]:
            score_col = f"{residual_name}_score"
            norm_col = f"{residual_name}_norm"
            th = threshold_map.get(residual_name, np.nan)

            df_eval[norm_col] = df_eval[score_col] / (th + 1e-12)
            df_eval[norm_col] = df_eval[norm_col].clip(lower=0, upper=5)

        # =========================
        # Yellow / Red mechanism confirmation
        # 黄色预警 / 红色确认
        # =========================
        # 机理一致性规则：扭矩异常必须出现，同时水力异常或钻进效率异常至少出现一个。
        # 时间滞后一致性：当前扭矩异常 + 最近一段时间内出现过水力或效率异常。
        sampling_seconds = 4
        lookback_points = max(
            1,
            int(args.confirm_lookback_minutes * 60 / sampling_seconds)
        )

        hyd_recent = (
            df_eval["R_hyd_alarm_raw"]
            .astype(int)
            .rolling(lookback_points, min_periods=1)
            .max()
            .astype(bool)
        )

        eff_recent = (
            df_eval["R_eff_alarm_raw"]
            .astype(int)
            .rolling(lookback_points, min_periods=1)
            .max()
            .astype(bool)
        )

        df_eval["R_torque_confirm_score"] = (
            df_eval["R_torque_score"]
            .rolling(window=max(1, int(60 / sampling_seconds)), min_periods=1)
            .median()
        )

        torque_confirm_threshold = args.torque_confirm_ratio * float(threshold_map["R_torque"])

        df_eval["R_torque_confirm_alarm_raw"] = (
            df_eval["R_torque_confirm_score"] > torque_confirm_threshold
        )

        rop_train = (
            df_train[COLS["rop"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(rop_train) > 0:
            rop_low_threshold = float(np.quantile(rop_train, args.rop_low_quantile))
        else:
            rop_low_threshold = np.nan

        df_eval["ROP_smooth"] = (
            df_eval[COLS["rop"]]
            .rolling(window=max(1, int(3 * 60 / sampling_seconds)), min_periods=1)
            .median()
        )

        df_eval["ROP_low_raw"] = df_eval["ROP_smooth"] < rop_low_threshold

        rop_low_recent = (
            df_eval["ROP_low_raw"]
            .astype(int)
            .rolling(lookback_points, min_periods=1)
            .max()
            .astype(bool)
        )

        df_eval["R_confirm_yellow_raw"] = (
            df_eval["R_torque_confirm_alarm_raw"]
            & (hyd_recent | eff_recent)
        )

        # df_eval["R_confirm_red_raw"] = (
        #     df_eval["R_torque_confirm_alarm_raw"]
        #     & hyd_recent
        #     & eff_recent
        #     & rop_low_recent
        # )
        # df_eval["R_confirm_red_raw"] = (
        #     df_eval["R_torque_confirm_alarm_raw"]
        #     & hyd_recent
        #     & (eff_recent | rop_low_recent)
        # )
        # =========================
        # Stricter red confirmation / 更严格的红色确认
        # =========================

        red_lookback_minutes = 3.0
        red_lookback_points = max(
            1,
            int(red_lookback_minutes * 60 / sampling_seconds)
        )

        # Strong torque condition:
        # 红色确认要求扭矩残差达到原始强阈值，而不是 0.6 倍中等阈值
        torque_strong = (
            df_eval["R_torque_confirm_score"] > float(threshold_map["R_torque"])
        )

        # Strong / recent hydraulic condition:
        # 要求最近短窗口内出现明显水力异常
        hyd_strong_recent = (
            (df_eval["R_hyd_norm"] > 1.0)
            .astype(int)
            .rolling(red_lookback_points, min_periods=1)
            .max()
            .astype(bool)
        )

        # Current efficiency degradation:
        # 不再用 15 分钟 eff_recent，改用最近 3 分钟内的效率异常
        eff_current = (
            df_eval["R_eff_alarm_raw"]
            .astype(int)
            .rolling(red_lookback_points, min_periods=1)
            .max()
            .astype(bool)
        )

        # Current low ROP:
        # 不再用 15 分钟 rop_low_recent，改用最近 3 分钟低 ROP
        rop_low_current = (
            df_eval["ROP_low_raw"]
            .astype(int)
            .rolling(red_lookback_points, min_periods=1)
            .max()
            .astype(bool)
        )

        df_eval["R_confirm_red_raw"] = (
            df_eval["R_confirm_yellow_raw"]
            & torque_strong
            & hyd_strong_recent
            & (eff_current | rop_low_current)
        )


        # Persistence requirement
        # 加连续点约束，避免单点尖峰误报
        df_eval["R_confirm_yellow"] = (
            df_eval["R_confirm_yellow_raw"]
            .astype(int)
            .rolling(window=args.persistence_points, min_periods=args.persistence_points)
            .sum()
            >= args.persistence_points
        ).fillna(False)

        df_eval["R_confirm_red"] = (
            df_eval["R_confirm_red_raw"]
            .astype(int)
            .rolling(window=args.persistence_points, min_periods=args.persistence_points)
            .sum()
            >= args.persistence_points
        ).fillna(False)

        df_eval["R_confirm_alarm_raw"] = df_eval["R_confirm_yellow_raw"]
        df_eval["R_confirm_alarm"] = df_eval["R_confirm_yellow"]

        # For visualization:
        # A continuous score showing the strength of the confirmed mechanism evidence.
        # 用连续分数辅助画图：扭矩证据权重大，水力和效率作为辅助证据。
        # Continuous score for visualization only.
        # 连续分数只用于画图，不作为最终报警规则。
        df_eval["R_confirm_score"] = (
            0.50 * df_eval["R_torque_norm"].fillna(0)
            + 0.25 * df_eval["R_hyd_norm"].fillna(0)
            + 0.25 * df_eval["R_eff_norm"].fillna(0)
        )

        # =========================
        # Summary for yellow warning
        # 黄色预警 summary
        # =========================
        yellow_first_time, yellow_warning_min, yellow_false_episodes = get_alarm_metrics(
            df=df_eval,
            time_col=args.time_col,
            alarm_col="R_confirm_yellow",
            event_time=event_time,
            no_stuck=args.no_stuck,
        )

        summary_rows.append(
            {
                "residual": "R_confirm_yellow",
                "meaning": "Yellow warning: R_torque_confirm & (R_hyd_recent | R_eff_recent)",
                "threshold": "rule_based_yellow",
                "first_alarm_time": yellow_first_time,
                "warning_min": yellow_warning_min,
                "false_alarm_episodes_before_60min": yellow_false_episodes,
                "train_rows": len(df_train),
                "eval_rows": len(df_eval),
            }
        )

        # =========================
        # Summary for red confirmation
        # 红色确认 summary
        # =========================
        red_first_time, red_warning_min, red_false_episodes = get_alarm_metrics(
            df=df_eval,
            time_col=args.time_col,
            alarm_col="R_confirm_red",
            event_time=event_time,
            no_stuck=args.no_stuck,
        )

        summary_rows.append(
            {
                "residual": "R_confirm_red",
                "meaning": "Red confirmation: yellow warning & strong torque & strong hydraulic & current efficiency/ROP degradation",
                "threshold": "rule_based_red",
                "first_alarm_time": red_first_time,
                "warning_min": red_warning_min,
                "false_alarm_episodes_before_60min": red_false_episodes,
                "train_rows": len(df_train),
                "eval_rows": len(df_eval),
            }
        )

        plot_start = event_time - pd.Timedelta(minutes=args.plot_minutes_before)
        plot_end = event_time + pd.Timedelta(minutes=args.plot_minutes_after)
        df_plot = df_eval[
            (df_eval[args.time_col] >= plot_start) & (df_eval[args.time_col] <= plot_end)
        ]

        # The threshold line here is only a visual reference:
        # R_confirm_score >= 1 roughly means the combined normalized mechanism evidence is strong.
        # 这里 1.0 只是画图参考线，不是最终报警规则；最终报警由上面的机理一致性规则决定。
        plot_score(
            df_plot=df_plot,
            time_col=args.time_col,
            score_col="R_confirm_score",
            threshold=1.0,
            event_time=event_time,
            output_path=output_dir / "R_confirm_score.png",
            title="R_confirm: mechanism-consistency confirmation",
        )

        print(f"[DONE] R_confirm_yellow: first_alarm={yellow_first_time}, warning_min={yellow_warning_min}")
        print(f"[DONE] R_confirm_red: first_alarm={red_first_time}, warning_min={red_warning_min}")

    else:
        print("[SKIP] R_confirm: required residual columns are missing.")
        # norm_cols = []
        # for name, score_col, threshold in available_scores:
        #     norm_col = f"{name}_norm"
        #     df_eval[norm_col] = df_eval[score_col] / (threshold + 1e-12)
        #     df_eval[norm_col] = df_eval[norm_col].clip(lower=0, upper=5)
        #     norm_cols.append(norm_col)

        # df_eval["R_fusion_score"] = df_eval[norm_cols].mean(axis=1)
        # fusion_threshold = 1.0

        # df_eval["R_fusion_alarm_raw"], df_eval["R_fusion_alarm"] = add_persistent_alarm(
        #     df_eval, "R_fusion_score", fusion_threshold, args.persistence_points
        # )

        # first_time, warning_min = first_alarm_before_event(
        #     df_eval, args.time_col, "R_fusion_alarm", event_time
        # )

        # false_region = df_eval[df_eval[args.time_col] < (event_time - pd.Timedelta(minutes=60))]
        # false_alarm_episodes = count_alarm_episodes(false_region["R_fusion_alarm"])

        # summary_rows.append(
        #     {
        #         "residual": "R_fusion",
        #         "meaning": "Average normalized mechanism residuals / 多机理残差平均融合",
        #         "threshold": fusion_threshold,
        #         "first_alarm_time": first_time,
        #         "warning_min": warning_min,
        #         "false_alarm_episodes_before_60min": false_alarm_episodes,
        #         "train_rows": len(df_train),
        #         "eval_rows": len(df_eval),
        #     }
        # )

        # plot_start = event_time - pd.Timedelta(minutes=args.plot_minutes_before)
        # plot_end = event_time + pd.Timedelta(minutes=args.plot_minutes_after)
        # df_plot = df_eval[
        #     (df_eval[args.time_col] >= plot_start) & (df_eval[args.time_col] <= plot_end)
        # ]

        # plot_score(
        #     df_plot=df_plot,
        #     time_col=args.time_col,
        #     score_col="R_fusion_score",
        #     threshold=fusion_threshold,
        #     event_time=event_time,
        #     output_path=output_dir / "R_fusion_score.png",
        #     title="R_fusion: average normalized mechanism residuals",
        # )

        # print(f"[DONE] R_fusion: first_alarm={first_time}, warning_min={warning_min}")
    
    

        # torque_threshold = threshold_map["R_torque"]

        # # Smooth torque residual to suppress isolated spikes
        # # 对扭矩残差做平滑，减少单点尖峰影响
        # smooth_points = max(1, int(60 / 4))  # 1 minute window, 4s sampling
        # df_eval["R_torque_smooth"] = (
        #     df_eval["R_torque_score"]
        #     .rolling(window=smooth_points, min_periods=1)
        #     .median()
        # )

        # df_eval["R_mech_alarm_raw"] = df_eval["R_torque_smooth"] > torque_threshold

        # # Suppress alarms during driller adjustment if mask exists
        # # 如果存在司钻调参屏蔽，则调参期不报警
        # if "driller_adjust_mask" in df_eval.columns:
        #     df_eval["R_mech_alarm_raw"] = (
        #         df_eval["R_mech_alarm_raw"] & (~df_eval["driller_adjust_mask"])
        #     )

        # # Persistence: require continuous mechanical abnormality
        # # 连续异常约束：要求机械异常持续一段时间
        # df_eval["R_mech_alarm"] = (
        #     df_eval["R_mech_alarm_raw"]
        #     .astype(int)
        #     .rolling(window=args.persistence_points, min_periods=args.persistence_points)
        #     .sum()
        #     >= args.persistence_points
        # ).fillna(False)

        # first_time, warning_min = first_alarm_before_event(
        #     df_eval, args.time_col, "R_mech_alarm", event_time
        # )

        # false_region = df_eval[df_eval[args.time_col] < (event_time - pd.Timedelta(minutes=60))]
        # false_alarm_episodes = count_alarm_episodes(false_region["R_mech_alarm"])

        # summary_rows.append(
        #     {
        #         "residual": "R_mech",
        #         "meaning": "Mechanical-dominant warning based on smoothed torque residual",
        #         "threshold": torque_threshold,
        #         "first_alarm_time": first_time,
        #         "warning_min": warning_min,
        #         "false_alarm_episodes_before_60min": false_alarm_episodes,
        #         "train_rows": len(df_train),
        #         "eval_rows": len(df_eval),
        #     }
        # )

        # plot_start = event_time - pd.Timedelta(minutes=args.plot_minutes_before)
        # plot_end = event_time + pd.Timedelta(minutes=args.plot_minutes_after)
        # df_plot = df_eval[
        #     (df_eval[args.time_col] >= plot_start)
        #     & (df_eval[args.time_col] <= plot_end)
        # ]

        # plot_score(
        #     df_plot=df_plot,
        #     time_col=args.time_col,
        #     score_col="R_torque_smooth",
        #     threshold=torque_threshold,
        #     event_time=event_time,
        #     output_path=output_dir / "R_mech_score.png",
        #     title="R_mech: smoothed torque-residual mechanical warning",
        # )

        # print(f"[DONE] R_mech: first_alarm={first_time}, warning_min={warning_min}")
    
    
    # =========================
    # Mechanical-dominant trend warning / 机械主导型趋势预警
    # =========================
    if "R_torque_score" in df_eval.columns and "R_torque" in threshold_map:
        torque_threshold = float(threshold_map["R_torque"])

        # Sampling interval is about 4 seconds.
        # 采样间隔约为 4 秒。
        sampling_seconds = 4

        short_minutes = 1.0
        long_minutes = 10.0
        warmup_minutes = 10.0

        short_points = max(1, int(short_minutes * 60 / sampling_seconds))
        long_points = max(short_points + 1, int(long_minutes * 60 / sampling_seconds))
        warmup_points = max(1, int(warmup_minutes * 60 / sampling_seconds))

        # Short-term smoothed torque residual.
        # 短窗口平滑扭矩残差。
        df_eval["R_torque_short"] = (
            df_eval["R_torque_score"]
            .rolling(window=short_points, min_periods=1)
            .median()
        )

        # Long-term historical baseline.
        # 长窗口历史基线。shift 是为了避免当前异常进入基线。
        df_eval["R_torque_long_base"] = (
            df_eval["R_torque_score"]
            .rolling(window=long_points, min_periods=1)
            .median()
            .shift(short_points)
        )

        df_eval["R_torque_long_base"] = df_eval["R_torque_long_base"].fillna(0.0)

        # Trend score: current smoothed residual minus historical baseline.
        # 趋势分数：当前短期残差相对于历史基线的抬升量。
        df_eval["R_mech_trend_score"] = (
            df_eval["R_torque_short"] - df_eval["R_torque_long_base"]
        ).clip(lower=0)

        # Compute trend threshold only from the normal training period.
        # 趋势阈值只从正常训练段计算，避免未来信息泄漏。
        train_trend = df_eval.loc[
            df_eval[args.time_col] < train_end_time,
            "R_mech_trend_score"
        ]

        trend_threshold = robust_threshold(
            train_trend,
            q=0.95,
            mad_k=args.mad_k
        )

        # Early mechanical warning:
        # 1) torque residual has risen above a moderate level;
        # 2) torque residual is significantly higher than its own historical baseline.
        #
        # 早期机械预警：
        # 1）扭矩残差达到中等强度；
        # 2）扭矩残差相对历史基线显著抬升。
        moderate_torque_level = 0.40 * torque_threshold

        df_eval["R_mech_alarm_raw"] = (
            (df_eval["R_torque_short"] > moderate_torque_level)
            & (df_eval["R_mech_trend_score"] > trend_threshold)
        )

        # Ignore initial unstable rolling-window period.
        # 忽略开头 rolling window 尚未稳定的阶段。
        df_eval.loc[df_eval.index < warmup_points, "R_mech_alarm_raw"] = False

        # Suppress alarms during active driller adjustments.
        # 司钻主动调参及稳定期内不报警。
        if "driller_adjust_mask" in df_eval.columns:
            df_eval["R_mech_alarm_raw"] = (
                df_eval["R_mech_alarm_raw"] & (~df_eval["driller_adjust_mask"])
            )

        # Persistence requirement.
        # 连续异常约束。
        df_eval["R_mech_alarm"] = (
            df_eval["R_mech_alarm_raw"]
            .astype(int)
            .rolling(window=args.persistence_points, min_periods=args.persistence_points)
            .sum()
            >= args.persistence_points
        ).fillna(False)

        # first_time, warning_min = first_alarm_before_event(
        #     df_eval, args.time_col, "R_mech_alarm", event_time
        # )

        # false_region = df_eval[
        #     df_eval[args.time_col] < (event_time - pd.Timedelta(minutes=60))
        # ]
        # false_alarm_episodes = count_alarm_episodes(false_region["R_mech_alarm"])
        first_time, warning_min, false_alarm_episodes = get_alarm_metrics(
            df=df_eval,
            time_col=args.time_col,
            alarm_col="R_mech_alarm",
            event_time=event_time,
            no_stuck=args.no_stuck,
        )

        summary_rows.append(
            {
                "residual": "R_mech_trend",
                "meaning": "Mechanical-dominant trend warning based on torque residual baseline shift",
                "threshold": trend_threshold,
                "first_alarm_time": first_time,
                "warning_min": warning_min,
                "false_alarm_episodes_before_60min": false_alarm_episodes,
                "train_rows": len(df_train),
                "eval_rows": len(df_eval),
            }
        )

        plot_start = event_time - pd.Timedelta(minutes=args.plot_minutes_before)
        plot_end = event_time + pd.Timedelta(minutes=args.plot_minutes_after)
        df_plot = df_eval[
            (df_eval[args.time_col] >= plot_start)
            & (df_eval[args.time_col] <= plot_end)
        ]

        plot_score(
            df_plot=df_plot,
            time_col=args.time_col,
            score_col="R_mech_trend_score",
            threshold=trend_threshold,
            event_time=event_time,
            output_path=output_dir / "R_mech_trend_score.png",
            title="R_mech_trend: torque-residual baseline shift warning",
        )

        print(f"[DONE] R_mech_trend: first_alarm={first_time}, warning_min={warning_min}")
        # =========================
    # Final red confirmation / 最终红色确认
    # =========================
    # Final red = yellow warning + recent mechanical trend confirmation
    # 最终红色确认 = 黄色预警 + 近期机械趋势确认
    if all(c in df_eval.columns for c in ["R_confirm_yellow", "R_mech_alarm"]):
        sampling_seconds = 4
        final_lookback_points = max(
            1,
            int(args.confirm_lookback_minutes * 60 / sampling_seconds)
        )

        mech_recent = (
            df_eval["R_mech_alarm"]
            .astype(int)
            .rolling(final_lookback_points, min_periods=1)
            .max()
            .astype(bool)
        )

        df_eval["R_final_red_raw"] = (
            df_eval["R_confirm_yellow"].astype(bool)
            & mech_recent
        )

        df_eval["R_final_red"] = (
            df_eval["R_final_red_raw"]
            .astype(int)
            .rolling(window=args.persistence_points, min_periods=args.persistence_points)
            .sum()
            >= args.persistence_points
        ).fillna(False)

        final_first_time, final_warning_min, final_false_episodes = get_alarm_metrics(
            df=df_eval,
            time_col=args.time_col,
            alarm_col="R_final_red",
            event_time=event_time,
            no_stuck=args.no_stuck,
        )

        summary_rows.append(
            {
                "residual": "R_final_red",
                "meaning": "Final red confirmation: R_confirm_yellow & recent R_mech_trend",
                "threshold": "rule_based_final_red",
                "first_alarm_time": final_first_time,
                "warning_min": final_warning_min,
                "false_alarm_episodes_before_60min": final_false_episodes,
                "train_rows": len(df_train),
                "eval_rows": len(df_eval),
            }
        )

        print(f"[DONE] R_final_red: first_alarm={final_first_time}, warning_min={final_warning_min}")
    else:
        print("[SKIP] R_final_red: R_confirm_yellow or R_mech_alarm missing.")
    
    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary_stage1.csv"
    residual_path = output_dir / "residuals_stage1.csv"
    print("R_confirm_score valid points:", df_eval["R_confirm_score"].notna().sum() if "R_confirm_score" in df_eval.columns else "missing")
    summary.to_csv(summary_path, index=False)
    df_eval.to_csv(residual_path, index=False)

    print("=" * 80)
    print(f"Summary saved to: {summary_path}")
    print(f"Residuals saved to: {residual_path}")
    print(f"Figures saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
