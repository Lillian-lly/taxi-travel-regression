"""统一的回归实验入口。参数选择仅使用验证集，测试集在设置冻结后评估。"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import ROOT, FEATURES, FEATURE_HELP, training_data, extract_features, read_pickle, open_data

SEED = 42
NAMES = {
    "fixed_rule": "固定点数公式", "point_linear": "单特征线性回归",
    "linear": "多特征线性回归", "rf_before": "随机森林优化前",
    "random_forest": "随机森林优化后", "lgb_before": "LightGBM优化前",
    "lightgbm": "LightGBM优化后", "linear_no_points": "线性回归去掉点数",
}


def score(y, pred):
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred)),
            "MAPE_percent": float(np.mean(np.abs((y - pred) / y)) * 100)}


def linear_model():
    # 中位数、均值和标准差只由 fit 接收到的训练数据学习。
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())


def build_model(kind, params, jobs):
    if kind == "random_forest":
        regressor = RandomForestRegressor(random_state=SEED, n_jobs=jobs,
                                         max_samples=0.85, max_leaf_nodes=4096, **params)
    else:
        from lightgbm import LGBMRegressor
        regressor = LGBMRegressor(random_state=SEED, n_jobs=jobs, verbosity=-1,
                                 deterministic=True, force_col_wise=True,
                                 subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.9, **params)
    # 保留列名传给树模型，预测与训练一致，也便于查看特征名称。
    return make_pipeline(SimpleImputer(strategy="median").set_output(transform="pandas"), regressor)


def tune(kind, candidates, X_train, y_train, X_val, y_val, jobs):
    """每个候选参数只拟合训练集；LGBM早停也只查看验证集。"""
    rows, chosen = [], None
    for i, params in enumerate(candidates):
        started = time.perf_counter()
        model = build_model(kind, params, jobs)
        if kind == "lightgbm":
            from lightgbm import early_stopping
            # 手动先拟合填补器，保证 eval_set 与训练端预处理一致。
            imputer, regressor = model.steps[0][1], model.steps[-1][1]
            a = imputer.fit_transform(X_train)
            b = imputer.transform(X_val)
            regressor.fit(a, y_train, eval_set=[(b, y_val)], eval_metric="mae",
                          callbacks=[early_stopping(80, first_metric_only=True, verbose=False)])
            actual = int(regressor.best_iteration_ or params["n_estimators"])
        else:
            model.fit(X_train, y_train)
            actual = params["n_estimators"]
        metrics = score(y_val, model.predict(X_val))
        final_params = dict(params, n_estimators=actual)
        row = {"model": kind, "candidate": i, "stage": "before" if i == 0 else "candidate",
               "params": json.dumps(params, ensure_ascii=False), "used_trees": actual,
               "fit_seconds": time.perf_counter() - started, **metrics}
        rows.append(row)
        print(f"  {kind} 候选{i}: 验证MAE={metrics['MAE']:.4f}s, 树数={actual}, 耗时={row['fit_seconds']:.1f}s", flush=True)
        if chosen is None or metrics["MAE"] < chosen["MAE"]:
            chosen = {"MAE": metrics["MAE"], "candidate": i, "params": final_params}
    for row in rows:
        row["selected"] = row["candidate"] == chosen["candidate"]
    before = dict(candidates[0], n_estimators=rows[0]["used_trees"])
    return chosen, before, rows


def make_figure(output, metrics, predictions, models):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    selected = [k for k in ["fixed_rule", "point_linear", "linear", "random_forest", "lightgbm"] if k in predictions]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    test = metrics[metrics.split == "test"].set_index("model")
    for ax, metric in zip(axes[0, :2], ["MAE", "RMSE"]):
        values = test.loc[selected, metric]
        bars = ax.barh([NAMES[k] for k in selected], values, color="#4478a8")
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_xlim(0, values.max() * 1.2)
        ax.set_xlabel("秒")
        ax.set_title(f"测试集 {metric}（越小越好）")
    index = np.random.default_rng(SEED).choice(len(predictions), min(2500, len(predictions)), replace=False)
    sample = predictions.iloc[index]
    y = sample.travel_time
    for kind in ["linear", "random_forest", "lightgbm"]:
        if kind in sample:
            axes[0, 2].scatter(y, sample[kind], s=5, alpha=0.3, label=NAMES[kind])
            axes[1, 0].hist(sample[kind] - y, bins=70, alpha=0.45, label=NAMES[kind])
    axes[0, 2].plot([y.min(), y.max()], [y.min(), y.max()], "k--", linewidth=1)
    axes[0, 2].set(title="真实值与预测值（固定抽样）", xlabel="真实耗时（秒）", ylabel="预测耗时（秒）")
    axes[0, 2].legend()
    axes[1, 0].set(title="误差分布（固定抽样）", xlabel="预测－真实（秒）", ylabel="行程数量")
    axes[1, 0].legend()
    for ax, kind in zip(axes[1, 1:], ["random_forest", "lightgbm"]):
        if kind not in models:
            ax.axis("off")
            continue
        reg = models[kind].steps[-1][1]
        importance = pd.Series(reg.feature_importances_, index=FEATURES).nlargest(7).sort_values()
        ax.barh([FEATURE_HELP[k] for k in importance.index], importance.values, color="#4e9478")
        label = "不纯度下降" if kind == "random_forest" else "分裂增益"
        ax.set_title(f"{NAMES[kind]}重要性（{label}，非因果）")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["all", "v1", "v2"], default="all")
    parser.add_argument("--output", default="results", help="相对task_B目录的输出位置，也支持绝对路径")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs 必须至少为1")
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("1/5 从原始训练数据提取无泄漏特征", flush=True)
    X, y = training_data(ROOT / "data_ds15/train.pkl")
    indices = np.arange(len(y))
    fit_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=SEED)
    train_idx, val_idx = train_test_split(fit_idx, test_size=0.15 / 0.85, random_state=SEED)
    assert not (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx))
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(y)
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    specs = {"point_linear": ("linear", ["point_count"], {}),
             "linear": ("linear", FEATURES, {}),
             "linear_no_points": ("linear", [f for f in FEATURES if f != "point_count"], {})}
    all_tuning, selection, validation_rows = [], {}, []
    print(f"2/5 验证集选择参数；train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}", flush=True)
    candidates = {}
    if args.version in ["all", "v1"]:
        candidates["random_forest"] = [
            {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 10},
            {"n_estimators": 160, "max_depth": 18, "min_samples_leaf": 3},
            {"n_estimators": 160, "max_depth": None, "min_samples_leaf": 8},
            {"n_estimators": 160, "max_depth": 18, "min_samples_leaf": 1},
        ]
    if args.version in ["all", "v2"]:
        candidates["lightgbm"] = [
            {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 8, "objective": "regression", "metric": "mae", "importance_type": "gain"},
            {"n_estimators": 1200, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 8, "objective": "mae", "metric": "mae", "importance_type": "gain"},
            {"n_estimators": 1200, "learning_rate": 0.03, "num_leaves": 63, "max_depth": 8, "objective": "mae", "metric": "mae", "importance_type": "gain"},
            {"n_estimators": 1000, "learning_rate": 0.03, "num_leaves": 31, "max_depth": 8, "objective": "regression", "metric": "mae", "importance_type": "gain"},
        ]
    for kind, options in candidates.items():
        chosen, before, rows = tune(kind, options, X_train, y_train, X_val, y_val, args.jobs)
        selection[kind] = chosen
        all_tuning.extend(rows)
        specs["rf_before" if kind == "random_forest" else "lgb_before"] = (kind, FEATURES, before)
        specs[kind] = (kind, FEATURES, chosen["params"])
        for row in rows:
            if row["candidate"] == 0:
                validation_rows.append({"split": "validation", "model": "rf_before" if kind == "random_forest" else "lgb_before",
                                        **{k: row[k] for k in score(y_val, y_val)}})
            if row["selected"]:
                validation_rows.append({"split": "validation", "model": kind, **{k: row[k] for k in score(y_val, y_val)}})
        pd.DataFrame(all_tuning).to_csv(output / "tuning.csv", index=False, encoding="utf-8-sig")
    for key, (_, columns, _) in specs.items():
        if key in ["point_linear", "linear", "linear_no_points"]:
            m = linear_model().fit(X_train[columns], y_train)
            validation_rows.append({"split": "validation", "model": key, **score(y_val, m.predict(X_val[columns]))})
    validation_rows.append({"split": "validation", "model": "fixed_rule", **score(y_val, 15 * (X_val.point_count.to_numpy() - 1))})
    print("3/5 参数已冻结；合并训练与验证数据，重新拟合最终模型", flush=True)
    models, model_features, fit_seconds = {}, {}, {}
    test_predictions = pd.DataFrame({"source_index": test_idx, "point_count": X.iloc[test_idx].point_count.to_numpy(), "travel_time": y[test_idx]})
    test_predictions["fixed_rule"] = 15 * (test_predictions.point_count - 1)
    # 旧题目数据仅在所有参数冻结后读取；不参与参数选择。
    old_records = read_pickle(ROOT / "task_B_tte/val_input.pkl")
    X_old = extract_features(old_records)
    truth = {int(t["traj_id"]): float(t["travel_time"]) for t in read_pickle(ROOT / "task_B_tte/val_gt.pkl")}
    old_ids = [int(t["traj_id"]) for t in old_records]
    if len(truth) != len(old_ids) or len(set(old_ids)) != len(old_ids) or set(truth) != set(old_ids):
        raise ValueError("旧验证输入和答案编号不完整或存在重复")
    y_old = np.array([truth[k] for k in old_ids])
    old_predictions = {"fixed_rule": 15 * (X_old.point_count.to_numpy() - 1)}
    for key, (kind, columns, params) in specs.items():
        begin = time.perf_counter()
        model = linear_model() if kind == "linear" else build_model(kind, params, args.jobs)
        model.fit(X.iloc[fit_idx][columns], y[fit_idx])
        test_predictions[key] = model.predict(X.iloc[test_idx][columns])
        old_predictions[key] = model.predict(X_old[columns])
        fit_seconds[key] = time.perf_counter() - begin
        print(f"  已完成 {NAMES[key]}，耗时{fit_seconds[key]:.1f}s", flush=True)
        if key not in ["rf_before", "lgb_before", "linear_no_points"]:
            models[key], model_features[key] = model, columns
        del model
    print("4/5 统一评估测试集并保存必要结果", flush=True)
    rows = list(validation_rows)
    for key in ["fixed_rule"] + list(specs):
        rows.append({"split": "test", "model": key, **score(y[test_idx], test_predictions[key].to_numpy())})
        rows.append({"split": "old_validation_reference", "model": key, **score(y_old, old_predictions[key])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    test_predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    versions = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib"]}
    if "lightgbm" in models:
        versions["lightgbm"] = importlib.metadata.version("lightgbm")
    with open_data(ROOT / "data_ds15/train.pkl") as training_file:
        training_sha256 = hashlib.file_digest(training_file, "sha256").hexdigest()
    metadata = {"version": args.version, "seed": SEED, "jobs": args.jobs,
                "n_samples": len(y), "missing_features": int(X.isna().sum().sum()),
                "features": FEATURES, "selection": selection,
                "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
                "split_indices": {"train": train_idx.tolist(), "validation": val_idx.tolist(), "test": test_idx.tolist()},
                "training_sha256": training_sha256,
                "versions": versions, "python": platform.python_version(), "final_fit_seconds": fit_seconds,
                "training_evaluation_seconds": time.perf_counter() - started}
    (output / "experiment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    default = "lightgbm" if "lightgbm" in models else "random_forest"
    joblib.dump({"models": models, "model_features": model_features, "default_model": default,
                 "features": FEATURES, "versions": versions}, output / "models.joblib", compress=3)
    print("5/5 生成模型对比与误差分析图", flush=True)
    make_figure(output, metrics, test_predictions, models)
    print(metrics[metrics.split == "test"].to_string(index=False), flush=True)
    print(f"完成：{output}；总耗时 {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
