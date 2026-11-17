# 出租车行程耗时回归实验

根据完整GPS轨迹和出发时间预测行程总耗时。版本一比较线性回归与随机森林，版本二使用LightGBM。两个版本共享14个特征、固定数据划分和评价指标。

方法、候选参数和结果解释见 [项目简要介绍](项目简要介绍.md)。

## 下载与运行

将仓库克隆或下载后，在VS Code中打开包含 `train.py` 的仓库根目录。推荐Python 3.11，以下为Windows PowerShell命令：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
```

仅运行一个版本并另存结果：

```powershell
.\.venv\Scripts\python.exe train.py --version v1 --output results_v1
.\.venv\Scripts\python.exe train.py --version v2 --output results_v2
```

默认最多使用8个线程，可加 `--jobs 4` 降低占用。训练会输出模型包、测试预测、评价表、参数搜索记录、配置与图表。

先完成默认训练，再加载模型预测：

```powershell
.\.venv\Scripts\python.exe predict.py
.\.venv\Scripts\python.exe predict.py --model linear
.\.venv\Scripts\python.exe predict.py --model random_forest
```

默认预测结果写入 `results/submission.csv`。可用 `--input`、`--bundle` 和 `--output` 指定其他输入、模型包和输出。

如果使用原课程工作区，仍可在上一级“期中大作业”目录运行 `.\.venv\Scripts\python.exe task_B/train.py`。

## 数据集

| 文件 | 内容 |
| --- | --- |
| `data_ds15/train.pkl.gz` | 132,657条训练行程，含坐标、时间戳等原始字段，无损gzip压缩 |
| `task_B_tte/val_input.pkl` | 原题16,582条验证行程的坐标及出发时间 |
| `task_B_tte/val_gt.pkl` | 对应的行程编号与真实耗时 |

程序直接读取压缩训练集，不需要手动解压。若本地已有原始 `train.pkl`，程序优先读取原文件，两者内容相同。未压缩训练集SHA256：

```text
2c5b7ea64e0703ac7d7f62c202869eccc7356127eda8c00bda0320e26c1588d9
```

实验从训练文件中按固定种子42划分92,859条训练、19,899条验证、19,899条测试行程。参数按验证MAE选择，再合并训练与验证集重训。原题验证数据仅作补充对照。旧的 `data_ds15/val.pkl` 与原题验证行程重复，因此没有重复上传。

## 代码与结果

| 文件 | 用途 |
| --- | --- |
| `features.py` | 提取轨迹几何、停顿、周期时间及中心区域等14个特征 |
| `train.py` | 数据划分、模型训练、验证调参、测试评估与绘图 |
| `predict.py` | 加载训练后的模型包进行预测 |
| `results/metrics.csv` | 验证、测试和原验证数据的评价指标 |
| `results/tuning.csv` | 候选参数及验证表现 |
| `results/comparison.png` | 模型效果、预测误差及特征重要性图 |

| 主要模型 | 测试MAE（秒） | 测试RMSE（秒） |
| --- | ---: | ---: |
| 多特征线性回归 | 20.95 | 30.68 |
| 随机森林选定配置 | 20.78 | 30.71 |
| LightGBM选定配置 | 19.72 | 30.21 |

大型模型包、逐条预测、环境目录、重复压缩包和运行截图不纳入仓库，可按上述命令生成。

数据沿用原课程项目整理的西安出租车轨迹。任务已知完整采样轨迹及点数，实验评价同一数据分布下的行程耗时预测。论文中请注明原课程资料提供的数据来源，并引用本仓库链接及所用提交版本。
