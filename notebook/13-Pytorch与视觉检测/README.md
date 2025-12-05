# Pytorch与视觉检测

张量

作用是代替 Numpy 库，类似于 Numpy 的多维数组

区别在于，Tensors 可以应用到 GPU 上加快计算速度

> PyTorch 提供了丰富的张量操作（如：加、减、乘、除、矩阵运算），适合高效处理数值计算。

---

PyTorch 是一个灵活、易用的深度学习框架，是研究和工业界的首选框架之一。

- 张量计算与自动微分
    - **核心数据结构是 Tensor**，支持 GPU 加速和自动求导（autograd），适合动态构建和调试模型。
- 动态计算图
    - **计算图在运行时动态生成，比静态图更直观**，便于调试和实现复杂模型（如：RNN、GAN）。
- 丰富的模块化设计
    - **torch.nn 听过预定义层（如：卷积、LSTM），torch.optim 包含优化器（如：SGD、Adam）**，可快速搭建神经网络。
- 分布式训练支持
    - 支持 **DataParallel（单机多卡）** 和 **DistributedDataParallel（多机多卡）**，结合 PyTorch Lightning 可进一步简化流程。

---





