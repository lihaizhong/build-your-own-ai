---
trigger: always_on
alwaysApply: true
---

1. 当前工程为 Python 工程，使用 uv 进行项目管理和运行。
2. 根目录下的 courseware 目录存放课程文件，这个目录下的文件仅可查阅，不可修改。
3. 根目录下的 experiment 目录是存放实验文件的目录，这个目录下的文件可修改。
4. 测试文件请使用 test__ 开头命名。

[代码规范]

选手需要提交能够针对A榜测试集产出预测结果的预测部分代码。所有文件请打包在zip压缩包内，对提交文件的组织形式要求如下：

1、原始数据文件夹 `data/`
选手无需提交天池提供的竞赛数据文件。

数据结构示例如下：

```plaintext
|-- data
    used_car_sample_submit.csv	
    used_car_train_20200313.csv	
    used_car_testA_20200313.csv
```

2、用户数据文件夹 `user_data/`
选手预测过程中需要生成的中间数据，请放入该文件夹中。文件夹下的子目录、文件名，选手可自行决定。

数据结构示例如下：

```plaintext
|-- user_data
    |-- tmp.dat
```

3、特征工程文件夹 `feature/`
数据处理和特征工程的代码，请放入该文件夹中。文件夹下的子目录、文件名，选手可自行决定。

数据结构示例如下：

```plaintext
|-- feature
    |-- generation.py
    |-- correlation.py
```

4、模型训练文件夹 `model/`
模型训练的代码，请放入该文件夹中。文件夹下的子目录、文件名，选手可自行决定。

数据结构示例如下：

```plaintext
|-- model
    |-- model.py
```

5、预测结果输出文件夹 `prediction_result/`
选手提交的代码，需要在此文件夹中产出针对A榜测试集的预测结果。预测结果文件的格式与竞赛中的提交要求一致，结果文件请命名为 `predictions.csv`。

数据结构如下：

```plaintext
|-- prediction_result
    |-- predictions.csv
```

6、代码文件夹 `code/`

- 请确保对A榜测试集的预测结果可以由提交的代码产出，预测流程所使用到的源码都要包含在提交的文件中。
- 所使用的依赖（操作系统版本，MATLAB/Python的版本，需要安装的Python package，使用到的TensorFlow，PyTorch，MXNet的版本等）都需要在requirements.txt文件中写明, 如‘lightgbm==2.3.1’。
- 如果使用深度神经网络，请提供网络定义等内容，确保能够产出预测结果
- 如果有需要编译的文件，请提供编译的脚本
- 请提供 `main.py` 或者 `main.sh` 文件作为程序入口，确保可以通过执行该文件来运行预测程序，得到最终结果，并将结果保存到上述的 `prediction_result/predictions.csv` 文件中
- 读入文件的路径请使用相对路径，比如 `../data/XX`

7、解决方案及算法介绍文件 `README.md`
请选手在 `README.md`（或其他文件格式如pdf等皆可）文件中

1. 介绍自己的解决方案及算法，包含从原始图像到最终结果输出的整个逻辑流程以及算法详情。
2. 代码运行说明，包括代码运行入口，若需额外输入参数请自行将参数写入运行代码。比如把 `python main.py param1 param2` 命令写进 `main.sh` 文件。
3. 若选手提交的代码在运行时有需要特殊注意的内容，也请在该文件中一并说明。

8、附注
提交代码文件夹结构举例

```plaintext
project
    |--README.md
    |--data
    |--user_data
    |--feature
    |--model
    |--prediction_result
        |—- predictions.csv
    |--code
        |-- main.py
        |-- requirements.txt
```

为了保证比赛的公平公正，如若满足下述任一条件，举办方都有权取消选手的参赛资格：

- 代码里面直接将预测结果写死在代码中作为提交结果的一部分输出；
- 无法复现结果，即代码运行失败、或复现结果与线上结果存在很大的差异；
- 存在核心代码抄袭的情况；
