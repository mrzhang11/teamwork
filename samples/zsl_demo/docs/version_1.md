**参考**：An Embarrassingly Simple Approach to Zero-Shot Learning

**算法框架**:  
![framework](https://github.com/mrzhang11/zsl_demo/blob/master/imgs/framework.png)

**具体步骤**：

（1）训练自编码器，对图像进行特征提取。这部分论文里没有涉及，可以尝试采用多种方法将原始数据映射到特征空间。

（2）将train/下的所有数据按照4:1的比例划分为训练集和验证集，二者的类别标签无交集。

（3）采用论文中带核函数的计算表达式，根据训练集学习权重V，利用V得到验证集的预测结果。

**存在的问题**：

（1）一些超参数没有调优；

（2）数据的获取和划分代码写的不够灵活；

（3）这个baseline， 结果太差了....acc:3%

（4）目前做的是conventional zero-shot，即训练集和测试集标签无交集。但是比赛中训练集的标签可能出现在测试集中，属于generalized zero-shot。

**可以做的事情**：

（1）算法：

	Generalized zero-shot;  
	    
	Another zero-shot baseline: semantic autoencoder for zero-shot learning；
	
	属性+词向量；
	
	图像增强（目前分辨率很低）；

（2）代码：

	灵活性和可复用性；
	计算效率；

（2）tricks：

	训练一个在训练集上表现非常好的分类器，通过提交代码判断测试集合中大概有多少是seen样本；
	模型ensemble；


