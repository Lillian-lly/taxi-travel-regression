"""不重训，不读取真实耗时，直接用保存的 Pipeline 预测行程。"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import ROOT, extract_features, read_pickle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "task_B_tte/val_input.pkl")
    parser.add_argument("--bundle", type=Path, default=ROOT / "results/models.joblib")
    parser.add_argument("--model", choices=["linear", "point_linear", "random_forest", "lightgbm"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/submission.csv")
    args = parser.parse_args()
    if not args.bundle.exists():
        parser.error(f"模型包不存在：{args.bundle}。请先运行 train.py，或用 --bundle 指定模型包。")
    bundle = joblib.load(args.bundle)
    name = args.model or bundle["default_model"]
    if name not in bundle["models"]:
        parser.error(f"模型包中没有{name}，已有：{list(bundle['models'])}")
    records = read_pickle(args.input)
    X = extract_features(records)
    columns = bundle["model_features"][name]
    pred = bundle["models"][name].predict(X[columns])
    if not np.isfinite(pred).all():
        raise ValueError("模型产生了非有限预测值")
    result = pd.DataFrame({"traj_id": [t.get("traj_id", i) for i, t in enumerate(records)], "predicted_time": pred})
    if result.traj_id.duplicated().any():
        raise ValueError("输入 traj_id 存在重复")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"使用 {name}，预测{len(result):,}条行程，保存至 {args.output}")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
