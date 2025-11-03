# 交互参考

## 课题描述

> 【AI入门系列】车市先知：二手车价格预测学习赛
> [https://tianchi.aliyun.com/competition/entrance/231784/information](https://tianchi.aliyun.com/competition/entrance/231784/information)
>
> 赛题以预测二手车的交易价格为任务，数据集报名后可见并可下载，该数据来自某交易平台的二手车交易记录，总数据量超过40w，包含31列变量信息，其中15列为匿名变量。
> 为了保证比赛的公平性，将会从中抽取15万条作为训练集，5万条作为测试集A，5万条作为测试集B，同时会对name、model、brand和regionCode等信息进行脱敏。

---

## 操作步骤（仅供参考）

### 01. 识别列信息

编写 Python，读取 **data** 目录中的 `used_car_train_20200313.csv` 文件的前5行数据，显示全部列信息。

### 02. 理解字段含义

理解上面列字段的含义，写入 Markdown 文件，放入 **docs** 目录下，名称为：`数据集说明.md`。

### 03. 制作EDA

帮我制作 EDA（Explorer Data Analysis），并将结果保存在 **docs** 目录下， 名称为：`探索性数据分析（EDA）报告.md`。

### 04. 数据预处理

通过 EDA 对 **data** 目录中的 `used_car_train_20200313.csv` 文件进行数据预处理，并告诉我你的建议，并将结果保存在 **docs** 目录下，名称为：`数据预处理建议.md`。不要写代码。

### 05. 数据特征处理

【手动处理】根据建议，整理特征信息，将不合理部分进行整理与规范。

数据特征处理，可以单独写一个 Python 文件，针对 **data** 目录中的 *训练集* 和 *测试集* 都进行特征预处理，并将预处理的结果进行保存，方便后续模型训练，将预处理结果保存在 **data** 目录下，训练集名称为：`used_car_train_preprocess.csv`， 测试集名称为：`used_car_testB_preprocess.csv`。
处理方式：
...

### 06. 模型训练

训练和测试的数据在 `user_data` 目录下。

以下是模型训练的方法，包括 **随机森林**、**XGBoost**、**LightGBM**、**CatBoost**和**模型融合**，请分别使用上述方法进行建模，并对测试集进行预测。

生成的模型统一放在 **model** 目录下，训练结果放在 **prediction_result** 目录下。

#### 随机森林建模

刚才我们做了这些特征预处理，现在使用 随机森林 进行建模，并对测试集进行预测。将预测结果写入到 `rf_result_[timestamp].csv` 文件中，将文件放到 **prediction_result** 目录下，生成出来的 `.kpl` 模型放到 **user_data** 目录下，名称为：`used_car_rf_model_[timestamp].kpl`。

---

#### XGBoost建模

帮我使用 XGBoost 进行建模，并对测试集进行预测。将预测结果写入到 `xgb_result_[timestamp].csv` 文件中，将文件放到 **prediction_result** 目录下，生成出来的 `.kpl` 模型放到 **user_data** 目录下，名称为：`used_car_xgb_model_[timestamp].kpl`。

---

#### LightGBM建模

帮我使用 LightGBM 进行建模，并对测试集进行预测。将预测结果写入到 `lgb_result_[timestamp].csv` 文件中，将文件放到 **prediction_result** 目录下，生成出来的 `.kpl` 模型放到 **user_data** 目录下，名称为：`used_car_lgb_model_[timestamp].kpl`。

---

#### CatBoost建模

帮我使用 CatBoost 进行建模，并对测试集进行预测。将预测结果写入到 `cat_result_[timestamp].csv` 文件中，将文件放到 **prediction_result** 目录下，生成出来的 `.kpl` 模型放到 **user_data** 目录下，名称为：`used_car_cat_model_[timestamp].kpl`。

---

#### 模型融合

帮我使用 模型融合 进行建模，并对测试集进行预测。将预测结果写入到 `fuse_result_[timestamp].csv` 文件中，将文件放到 **prediction_result** 目录下，生成出来的 `.kpl` 模型放到 **user_data** 目录下，名称为：`used_car_fuse_model_[timestamp].kpl`。

---
