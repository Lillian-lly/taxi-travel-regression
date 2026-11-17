"""把一趟 GPS 行程变成一行特征。只读坐标和出发时刻，不读终点时刻。"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import gzip
import pickle

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LOCAL_TZ = timezone(timedelta(hours=8))  # 显式使用中国时间，不随电脑时区改变
FEATURES = [
    "point_count", "total_distance", "straight_distance", "tortuosity",
    "stop_count", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "is_rush_hour", "is_night", "is_weekend", "start_in_center", "end_in_center",
]
FEATURE_HELP = {
    "point_count": "GPS 点数", "total_distance": "轨迹总距离（米）",
    "straight_distance": "起终点直接距离（米）", "tortuosity": "轨迹距离/起终点距离",
    "stop_count": "相邻点移动不足5米的片段数",
    "hour_sin": "出发时刻的正弦编码", "hour_cos": "出发时刻的余弦编码",
    "weekday_sin": "星期的正弦编码", "weekday_cos": "星期的余弦编码",
    "is_rush_hour": "是否7—9点或17—19点", "is_night": "是否23—5点",
    "is_weekend": "是否周末", "start_in_center": "起点是否距钟楼不足2公里",
    "end_in_center": "终点是否距钟楼不足2公里",
}


def open_data(path):
    """本地可读原始pkl；仓库提供等价的pkl.gz，直接解压读取。"""
    path = Path(path)
    if not path.exists():
        path = path.with_suffix(path.suffix + ".gz")
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def read_pickle(path):
    with open_data(path) as f:
        return pickle.load(f)


def distance(a, b):
    """Haversine 球面距离；支持单个点和多行坐标。"""
    a, b = np.radians(a), np.radians(b)
    d = b - a
    h = np.sin(d[..., 1] / 2) ** 2 + np.cos(a[..., 1]) * np.cos(b[..., 1]) * np.sin(d[..., 0] / 2) ** 2
    return 6371000.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def extract_one(traj):
    coords = np.asarray(traj["coords"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2:
        raise ValueError("每条轨迹必须至少包含两个[经度, 纬度]点")
    if not np.isfinite(coords).all():
        raise ValueError("轨迹坐标存在 NaN/Inf；Task B 需要完整路径")
    if (np.abs(coords[:, 0]) > 180).any() or (np.abs(coords[:, 1]) > 90).any():
        raise ValueError("坐标超出合法经纬度范围")
    if "departure_timestamp" in traj:
        departure = traj["departure_timestamp"]
    else:
        departure = traj["timestamps"][0]  # 只允许使用出发时刻
    dt = datetime.fromtimestamp(int(departure), tz=LOCAL_TZ)
    hour = dt.hour + dt.minute / 60 + dt.second / 3600
    weekday = dt.weekday()
    steps = distance(coords[:-1], coords[1:])
    direct = float(distance(coords[0], coords[-1]))
    total = float(steps.sum())
    center = np.array([108.940174, 34.258277])
    return {
        "point_count": len(coords), "total_distance": total,
        "straight_distance": direct,
        # 起终点几乎重合时不让分母趋于0；不按数据分布拟合阈值。
        "tortuosity": total / max(direct, 10.0),
        "stop_count": int((steps < 5.0).sum()),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7),
        "is_rush_hour": int(7 <= dt.hour <= 9 or 17 <= dt.hour <= 19),
        "is_night": int(dt.hour >= 23 or dt.hour <= 5),
        "is_weekend": int(weekday >= 5),
        "start_in_center": int(distance(coords[0], center) < 2000),
        "end_in_center": int(distance(coords[-1], center) < 2000),
    }


def extract_features(records):
    rows = []
    for i, traj in enumerate(records):
        if i % 30000 == 0:
            print(f"  提取特征 {i:,}/{len(records):,}", flush=True)
        try:
            rows.append(extract_one(traj))
        except (ValueError, KeyError, IndexError) as error:
            raise ValueError(f"第{i}条行程无法处理：{error}") from error
    return pd.DataFrame(rows, columns=FEATURES)


def training_data(path):
    records = read_pickle(path)
    X = extract_features(records)
    y = np.array([float(t["timestamps"][-1]) - float(t["timestamps"][0]) for t in records])
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("训练标签应为有限且大于0的总耗时")
    return X, y
