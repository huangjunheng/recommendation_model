# pytroch recommendation_system
想练习下用pytorch来复现下经典的推荐系统模型

1 实现了MF(Matrix Factorization, 矩阵分解)，在movielen 100k数据集上mse为0.853左右

2 实现了FM(Factorization machines, 因子分解机), 在movielen 100k数据集上mse为0.852
(只使用u, i, ratings 三元组的因子分解机与mf其实是一样的， 故在相同数据集上的结果也差不多）

参考论文：Steffen Rendle, Factorization Machines, ICDM2010.

3 DeepConn是第一篇使用深度学习模型利用评论信息进行推荐的文章，后面有很多改进工作如Transnets(ResSys2017), NARRE(www2018)等
所以说，这是篇非常值得认真阅读和复现的论文。

数据集下载地址：http://jmcauley.ucsd.edu/data/amazon/ 

使用的预训练文件下载地址：https://code.google.com/archive/p/word2vec/  下载GoogleNews-vectors-negative300.bin 文件放入data/embedding_data中

使用方法：1. 运行processing_data.py文件 2. 运行train文件

实验结果(mse)： office: 0.777, video_game: 1.182

参考论文：L.zheng et al, Joint deep modeling of users and items using reviews for recommendation. WSDM2017.

