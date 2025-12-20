# 分析式AI基础

训练赛：kaggle、天池

**机器学习** 是实现AI的一种方法，**AI大模型** 是机器学习能力的延伸。通过强大的算力和数据处理能力，使AI能够更广泛地应用于不同领域。

人工智能包含了 **机器学习**、**深度学习**、**强化学习**以及**大模型** 等。

---

## 分析式AI与生成式AI的区别

|     | 分析式（Analytical AI） | 生成式（Generative AI） |
| --- | --- | --- |
| 目标  | 分析和解释数据，提供洞察、预测或**决策支持**。 | **生成新内容**。这些内容在形式和风格上与训练数据相似。 |
| 用途  | 数据分析、预测建模、风险评估、分类任务等。 | 文本生成、图像创作、音频生成、内容创作等。 |
| 输出  | 报告、图表、预测结果、决策建议等。 | 新的文本、图像、音频、视频等内容。 |
| 技术原理 | 基于**统计分析**、**机器学习算法**（如回归分析、决策树、神经网络等） | 基于**生成式大模型**，如DeepSeek、Qwen、ChatGPT（Transformer架构）。 |
| 数据需求 | **需要高质量**、**标注清晰的数据**，以学习数据中的规律。 | **需要大量未标注数据**，以学习数据分布。 |
| 应用场景 | 金融风险评估、医疗诊断、市场预测、客户行为分析等。 | 广告设计、内容创作、游戏开发、艺术创作等。 |
| 风险挑战 | **模型可能过度拟合训练数据**，导致在新数据上表现不佳；数据偏差可能导致错误决策。 | 生成内容可能包含错误、误导性信息或不符合伦理的内容需防止内容被滥用。 |

---

## 十大经典机器学习算法

- 分类算法：C4.5，朴素贝叶斯（Naive Bayes），SVM，KNN，AdaBoost，CART

- 聚类算法：K-Means，EM

- 关联分析：Apriori

- 连接分析：PageRank

---

| 算法  | 工具  |
| --- | --- |
| 决策树 | from sklearn.tree import DecisionTreeClassifier |
| 朴素贝叶斯 | from sklearn.naive\_bayes import MultinomiaNB |
| SVM | from sklearn.svm import SVC |
| KNN | from sklearn.neighbors import KNeighborsClassifier |
| AdaBoost | from sklearn.ensemble import AdaBoostClassifier |
| K-Means | from sklearn.cluster import KMeans |
| EM  | from sklearn.mixture import GMM |
| Apriori | from efficient\_apriori import apriori |
| PageRank | import networkx as nx |

---

## 分类与回归

**分类** 与 **回归** 是监督学习中的**两大核心任务**，二者既有联系又有区别。

### 定义

**分类问题指的是将输入数据分配到预定义的离散类别中**。常见的分类任务包括垃圾邮件检测、手写数字识别等。

**分类的特征**：

- **输出为离散值**：即数据属于某个特定类别。
- **常用算法**：如逻辑回归、支持向量机、决策树、随机森林（RF）、K近邻（KNN）等。

**回归问题指的是预测一个连续的数值输出**。常见的回归任务包括房价预测、股票价格预测等。

**回归的特征**：

- **输出为连续值**：即预测结果为一个具体的数值。
- **常用算法**：如线性回归、决策树回归、支持向量回归（SVR）、Lasso回归等。

### 区别

- **输出类型**：**分类** 预测离散类别标签（如 [男, 女]或者多类别[猫、狗、鸟]）；**回归** 预测连续数值（如预测房价为15000元、预测资产金额为200万）。
- **评估指标**：**分类** 常用指标有 **准确率（Accuracy）**、**混淆矩阵**、**ROC曲线**、**F1分数** 等；**回归** 常用指标有 **均方误差（MSE）**、**均方根误差（RMSE）**、**R2** 等。
- **模型不同**：**分类** 常用的模型如 **逻辑回归**、**KNN**、**支持向量机** 等；**回归** 常用的模型如 **线性回归**、**岭回归**、**支持向量回归** 等。
- **目标函数**：**分类** 常用交叉熵损失（Cross-Entropy）；**回归** 常用均方误差（MSE）。

### 常用分类算法

#### K 近邻算法（K-Nearest Neighbors， KNN）

KNN 时一种基于距离的分类算法，通过找到与输入数据最近的 K 个样本来进行分类。

```python
from sklearn.neighbors import KNeighborsClassifier
```

#### 支持向量机（SVM）

SVM 是一种分类算法，它通过找到一个超平面，将数据点划分到不同的类别中。

```python
from sklearn.svm import SVC
```

### 常用回归算法

#### 决策树回归（Decision Tree Regressor）

决策树是一种基于树形结构的回归算法，通过递归划分特征空间来预测目标值。

```python
from sklearn.tree import DecisionTreeRegressor
```

#### 支持向量回归（SVR）

SVR 是支持向量机的回归版本，通过找到一个使得预算误差最小的超平面来进行回归预测。

```python
form sklearn.svm import SVR
```

### 如何选择分类/回归算法

1. **数据的输出类型**：首先根据输出是离散值还是连续值选择分类或回归算法
2. **数据的规模与维度**：不同的算法对数据 **规模** 和 **维度** 有不同的处理效果，如 **SVM** 适用于高维数据，而 **线性回归** 适用于低维数据。
3. **计算资源**：一些复杂的算法如 **支持向量机** 和 **神经网络** 需要大量计算资源，而简单的模型如 **线性回归** 和 **KNN** 相对较快。

---

## 贝叶斯思想

解决“**逆向概率**”问题的理论。

尝试解答在没有可靠证据的情况下，怎样做出更符合数据逻辑的推测。

**正向概率** 比较像上帝视角，即了解了事情的全貌再做判断。

**逆向概率** 则是从实际场景出发，预估时间概率。

比如，在不知道袋子中的黑球和白球的比例的前提下，如何通过摸出来的球颜色，判断袋子中黑白球的比例。

这套理论建立在主观判断的基础上：在我们不了解所有客观事实的情况下，同样可以先估计一个值。

- **先验概率**：通过经验来判断事情发生的概率。
- **后验概率**：就是发生结果之后，推测原因的概率。
- **条件概率**：P(A|B) 表示在事件 B 已经发生的前提下，事件 A 发生的概率。
- **似然函数**：把概率模型的训练过程理解为求参数估计的过程。 经过推导，贝叶斯推理告诉我们 **后验概率** 是与 **先验概率** 和 **似然函数** 成正比的。

---

![Pasted image 20250928053251.png](../../public/Pasted_Image_20250928053251.png)

- **高斯朴素贝叶斯**：特征变量是**连续变量**，符合高斯分布（正态分布），比如说人的身高，物体的长度。
- **多项式朴素贝叶斯**：特征变量是**离散变量**，符合多项分布，在文档分类中特征变量体现在一个单词出现的次数，或者是单词的TF-IDF值等。
- **伯努利朴素贝叶斯**：特征变量是**布尔变量**，符合0/1分布，在文档分类中特征是单词是否出现。

---

## 决策树思想

```python
from sklearn.tree import DecisionTreeClassifier

DecisionTreeClassifier(criterion="entropy")
```

参数

- criterion: 默认是基尼系数（`gini`），还可以是信息熵（`entropy`）。
- max\_depth: 决策树的最大深度，默认是 `None`。
- min\_samples\_split: 内部节点再划分所需最小样本数。
- max\_leaf\_nodes: 最大叶子节点。
- class\_weight: 类别权重。

## 随机森林思想

```python
from sklearn.ensumble import RandomForestClassifier

RandomForestClassifier(n_estimators=10, criterion="gini")
```

参数

- n\_estimators: 森林中决策树的个数，默认为 `10`。

- max\_features: 寻找最佳分割时需要考虑的特征数目，默认为 `auto`，即 `max_features=sqrt(n_features)`。

- criterion: 默认是基尼系数（`gini`），还可以是信息熵（`entropy`）。

- max\_depth: 决策树最大深度，默认为 `None`。

- min\_samples\_split: 内部节点再划分所需最小样本数。

- max\_leaf\_nodes: 最大叶子节点数。

- class\_weight: 类别权重。

---

## SVM思想

一些线性不可分的问题是可以非线性可分的，也就是在高维空间中存在分离超平面（separating hyperplane）。

使用非线性函数从原始的特征空间映射至更高维的空间，转化为线性可分问题。

![Pasted image 20250929031538.png](../../public/Pasted_Image_20250929031538.png)

`sklearn`中支持向量分类主要有三种方法：**SVC**、**NuSVC**、**LinearSVC**。

```python
"""Support Vector Classification 支持向量机用于分类"""
from sklearn.svm import SVC

SVC(
 C=1.0,
 kernel="rbf",
 degree=3,
 gamma="auto",
 coef0=0.0,
 shrinking=True,
 probability=False,
 tol=0.001,
 cache_size=200,
 class_weight=None,
 verbose=False,
 max_iter=-1,
 decision_function_shape="ovr",
 random_state=None
)

"""Nu-Support Vector Classification 核支持向量分类"""
from sklearn.svm import NuSVC

NuSVC(
 nu=0.5,
 kernel="rbf",
 degree=3,
 gamma="auto",
 coef0=0.0,
 shrinking=True,
 probability=False,
 tol=0.001,
 cache_size=200,
 class_weight=None,
 verbose=False,
 max_iter=-1,
 decision_function_shape="ovr",
 random_state=None
)

"""Linear Support Vector Classification 线性支持向量分类"""
from sklearn.svm import LinearSVC

LinearSVC(
 penalty="l2",
 loss="squared_hinge",
 dual=True,
 tol=0.0001,
 C=1.0,
 multi_class="ovr",
 fit_intercept=True,
 intercept_scaling=1,
 class_weight=None,
 verbose=0,
 random_state=None,
 max_iter=1000
)
```

参数

- C: 惩罚系数，类似于LR中的正则化系数，C越大惩罚越大。
- nu: 代表训练集训练的错误率的上限（用于 `NuSVC`）。
- kernel: 核函数类型，RBF，Linear，Poly，Sigmoid，precomputed，默认为RBF径向基核（高斯核函数）。
- gamma: 核函数系数，默认为 `auto`。
- degree: 当指定 kernel 为 `poly` 时，表示选择的多项式的最高次数，默认为 **三次多项式**。
- probability: 是否使用概率估计。
- shrinking: 是否进行启发式，SVM 只用少量训练样本进行计算。
- penalty: 正则化参数，**L1** 和 **L2** 两种参数可选（用于 LinearSVC）。
- loss: 损失函数，有 “hinge” 和 “squared\_hinge”两种可选，前者又称 L1损失，后者又称 L2损失。
- tol: 残差收敛条件，默认是 0.0001，与 LR 中的一致。

---

- SVC，Support Vector Classification，支持向量机用于分类
- SVR，Support Vector Regression，支持向量机用于回归

sklearn 中支持向量分类主要有三种方法：SVC、NuSVC、LinearSVC

- SVC，C-Support Vector Classification 支持向量分类
- NuSVC，Nu-Support Vector Classification 核支持向量分类（和SVC类似，不同的是可以使用参数来控制支持向量的个数）
- LinearSVC，Linear Support Vector Classification 线性支持向量分类（使用的核函数是linear）

Kernel 核的选择技巧：

- 如果样本数量 < 特征数量：
  - 方法1: 简单的使用 **线性核** 就可以，不用选择非线性核。
  - 方法2: 可以先对数据 **进行降维**，然后使用非线性核。
- 如果样本数量 > 特征数量：
  - 可以使用非线性核，将样本映射到更高维度，可以得到比较好的结果。

---

## 机器学习

机器学习包括 **监督学习**、**无监督学习**、**半监督学习**、**强化学习**。

- 符号学派，认为事情都是有因果的，机器可以自己摸索出规律，典型代表为决策树

- 贝叶斯学派，因果之间不是必然发生，是有一定概率的，即P(A|B)，典型代表为朴素贝叶斯

- 类推学派，通过类比可以让我们学习到很多未知的知识，所以我们需要先定义“相似度”，通过相似度进行发现

- 联结学派，模仿人脑神经元的工作原理，所有模式识别和记忆建立在神经元的不同连接方式上，典型代表为神经网络，深度学习

- 进化学派，上帝通过基因选择来适者生存，典型代表为遗传算法

![Pasted image 20250929040929.png](../../public/Pasted_Image_20250929040929.png)

---

XGBoost、LightGBM 和 CatBoost 是三种流行的机器学习算法，它们都是基于 **梯度提升决策树（GBDT）** 框架，但在工程优化和算法设计上各有侧重，旨在提高模型精度、训练速度和处理特定数据类型（如类别特征）的能力。

![Pasted image 20250924030429.png](../../public/Pasted_Image_20250924030429.png)

- XGBoost 以其卓越的准确性和丰富的工程优化著称。
- LightGBM 以其极快的训练速度和低内存占用为目标。
- CatBoost 通过创新的方法高效地处理类别特征，减少过拟合，并提供高性能的模型。

---

### XGBoost（Extreme Gradient Boosting）

#### 核心优势

在保持高准确性的同时，通过工程和算法优化提升了效率，是工业界广泛使用的梯度提升算法。

#### 关键技术

- 工程优化：支持分布式计算、改进缓存命中率。
- 并行和分块计算：通过分块和并行计算提高效率。
- 缺失值处理：对缺失值有内置处理机制。

---

### LightGBM（Light Gradient Boosting Machine）

#### 核心优势

专注于提升计算效率和降低内存占用，显著加快训练速度。

#### 关键技术

- Histogram-based 算法：使用直方图算法进行特征分桶，加快查找分裂点。
- Leaf-wise 策略：采用叶子节点生长策略，每次选择最优叶子分裂，比 Level-wise 策略（XGBoost使用）更快收敛，但需限制树的深度防止过拟合。
- GOSS（Gradient-based One-Side Sampling）：在查找分裂时过滤数据，保留了梯度值较大的样本。

---

### CatBoost（Categorical Boosting）

#### 核心优势

专门设计用来搞笑处理类别特征，并在准确性、泛化能力和处理类别特征方面表现优异。

#### 关键技术

- 对称决策树（Oblivious Trees）：使用对称的书结构作为基学习器，具有正则化作用并加快推断速度。
- Ordered Target Statistics：一种创新的方法，将类别特征转化为树脂特征，并利用先验值来减少样本数量波动造成的偏差。
- 组合类别特征：通过结合类别特征来挖掘它们之间的关联，丰富特征维度。
- 排序提升（Ordered Boosting）：一种对抗策略，可以有效处理数据集中的噪声点，防止预测偏移，提高模型的泛化能力。

---

## 总结

- XGBoost在传统机器学习领域仍然是最常用的算法之一，特别是在结构化数据的分类、回归和排序任务中表现突出。
- LightGBM在大规模数据集和高维度数据上表现更佳，适用于处理文本分类、图像分类、推荐系统等领域的数据。
- CatBoost在处理类别特征和缺失值方面表现出色，适用于电商推荐、医疗预测、金融风控等领域的数据。
- [Boosting三巨头：XGBoost、LightGBM和CatBoost](https://blog.csdn.net/qq_41667743/article/details/129417794)
- [CatBoost、LightGBM和XGBoost：谁才是最强王者！](https://zhuanlan.zhihu.com/p/654304590)
- [XGBoost、LightGBM、CatBoost](https://mayuanucas.github.io/xgboost-lightgbm/)
