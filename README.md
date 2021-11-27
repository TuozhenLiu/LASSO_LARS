# LASSO_LARS
An implement for LASSO by LARS Algorithm.

## Overview

- 针对LARS算法，本库在计算系数变化路径的基础上，实现了任给一个惩罚系数，通过系数分段线性变化的特点，得到系数估计。
- 进一步，可以任给一组惩罚系数，通过交叉验证，得到最优惩罚系数及其对应的系数估计。
- 此外，还提供了是否拟合截距项、是否标准化的功能。
- 上述算法、功能以sklearn-style编写成了自定义类LARS，并通过prostate数据进行应用示范。

- LARS类源程序写于 `LARS.py` ，相关自定义函数写于 `./src/*.py` 。
- LARS类的功能特点介绍、应用实例、及与sklearn包的对比写于 `LARS_User_Guide.ipynb` 。


## 1. Introduction of LASSO

&emsp; Lasso全称Least absolute shrinkage and selection operator，是一种以最小化绝对值（1范数）为惩罚函数的压缩估计方法，可以将变量的系数进行压缩并使某些回归系数变为0，进而达到变量选择的目的。Lasso方法可以应用于许多回归当中，对于多元线性回归，系数估计表达式可写成：

$$\begin{align}
  & \hat{\beta }(\lambda )=\arg \min \frac{1}{2}\parallel Y-X\beta \parallel _{2}^{2}+\lambda \parallel \beta {{\parallel }_{1}} \\ 
 & \hat{\beta }(\alpha )=\arg \min \frac{1}{2n}\parallel Y-X\beta \parallel _{2}^{2}+\alpha \parallel \beta {{\parallel }_{1}}  
\end{align}$$

&emsp; 其中$\alpha=\frac{\lambda}{2n}$，消除了样本量的影响，下文将采用$\alpha$作为惩罚系数。

&emsp; 对于多元线性回归的Lasso方法，常见的解法有最小角回归法（LARS）、坐标轴下降法：

1. 最小角回归法，通过系数$\hat{\beta }(\alpha )$分段线性变化的特点，计算其变化规律，然后再通过代入相应的惩罚系数$\alpha$来计算系数估计。


2. 坐标轴下降法，通过类似梯度下降法的启发式算法一步步迭代求解函数的最小值。

&emsp; 在给定一个$\alpha$时，最小角回归法需要计算全局的变化规律，计算量较大。当给定若干$\alpha$时，最小角回归法，只需计算一次路径变化规律，然后代入$\alpha$即可，坐标轴下降法需要若干次迭代过程。可见，当需要展示$\hat{\beta }(\alpha )$变化规律，或者从若干$\alpha$中寻找最优值时，最小角回归法有一定的优势。


## 2. Disadvantage of Sklearn
&emsp; sklearn包中主要有以下几个类解决多元线性回归的Lasso问题：*Lasso*、*LARS*、*LassoCV*、*lars_path* 

1. *Lasso*：通过坐标轴下降法、LARS，计算一个$\alpha$下系数估计结果，无法同时拟合不同$\alpha$下的多组系数。

2. *LassoCV*：通过坐标下降法，由内置算法给出交叉验证下最优的$\alpha$，但无法自定义一组$\alpha$去比较优劣。

3. *LARS*和*lars_path*：通过最小角回归法，给出$\hat{\beta }(\alpha )$的变化路径，但无法计算任一$\alpha$下系数估计结果。并且，在最小角回归的过程中算法存在错误，只包含变量进入过程，缺少变量退出过程。


&emsp; 此外，在上述类的初始化时，只能选择是否对x规范化，而不是标准化，也不能对y进行规范化或标准化操作。

## 3. Our LARS class
&emsp; 自定义LARS类的API与sklearn风格类似，融合了以上几个类的优点。具体介绍如下：

Parameters:	

* x：&emsp;&emsp; &emsp; &emsp;  &emsp; &emsp; np.array 自变量 nxp维
* y：&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; np.array 因变量 n维
* fit_intercept：&emsp; &emsp; &emsp; bool 是否在拟合截距项，默认为True
* standarize：&emsp; &emsp; &emsp; bool 是否在拟合前对x、y进行规范化，默认为True

Methods:

* get_path(plot=False)：&emsp; &emsp;计算$\hat{\beta }(\alpha )$和$\alpha$的变化路径，plot = True则输出系数变化规律图
* fit(alpha)：&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;计算给定alpha下的系数估计，alpha为float或list
* cv_fit(self, alpha, k=10)：&emsp;计算给定一组$\alpha$下，交叉验证后的最优$\alpha$及其$\hat{\beta }(\alpha )$，k为交叉验证折数
* predict(tx, coef)：&emsp; &emsp;&emsp;&emsp;计算给定预测集x和$\hat{\beta }(\alpha )$下的预测结果


Attributes(Public):

* alphas：&emsp; &emsp; &emsp; np.array $\alpha$的变化路径
* coefs：&emsp; &emsp; &emsp; np.array $\hat{\beta }(\alpha )$的变化路径
* coef_：&emsp; &emsp; &emsp; np.array 给定$\alpha$下的系数估计
* cv_coef：&emsp; &emsp; &emsp; np.array 给定一组$\alpha$下，最优系数估计
