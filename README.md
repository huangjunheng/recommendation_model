# mf_pytorch
想练习下用pytorch来复现下经典的推荐系统模型

1 实现了MF(Matrix Factorization, 矩阵分解)，在movielen 100k数据集上mse为0.853左右

2 实现了FM(Factorization machines, 因子分解机), 在movielen 100k数据集上mse为0.852
(只使用u, i, ratings 三元组的因子分解机与mf相同， 故在相同数据集上的结果也差不多）
参考论文：Steffen Rendle, Factorization Machines, ICDM2010

