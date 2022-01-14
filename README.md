# pytorch recommendation_system
想练习下用pytorch来复现下经典的推荐系统模型

1 实现了MF(Matrix Factorization, 矩阵分解)，在movielen 100k数据集上mse为0.853左右

2 实现了FM(Factorization machines, 因子分解机), 在movielen 100k数据集上mse为0.852
(只使用u, i, ratings 三元组的因子分解机与mf其实是一样的， 故在相同数据集上的结果也差不多）

参考论文：Steffen Rendle, Factorization Machines, ICDM 2010.

3 DeepConn是第一篇使用深度学习模型利用评论信息进行推荐的文章，后面有很多改进工作如Transnets(ResSys2017), NARRE(www2018)等
所以说，这是篇非常值得认真阅读和复现的论文。

数据集下载地址：http://jmcauley.ucsd.edu/data/amazon/ 

使用的预训练文件下载地址：https://code.google.com/archive/p/word2vec/  下载GoogleNews-vectors-negative300.bin 文件放入data/embedding_data中

使用方法：1. 运行pre_precessing.py文件 2. 运行train文件

实验结果(mse)： office: 0.777, video_game: 1.182

参考论文：L.zheng et al, Joint deep modeling of users and items using reviews for recommendation, WSDM 2017.

4 MMOE是谷歌于2018年提出的一种多任务学习推荐模型，被用于YouTube视频推荐场景，效果良好，被业界广泛关注，我在实习时线上模型也采用了MMOE的改进模型，叫PLE。
我在census数据集上进行实验，实验结果比原文差一点。

实验结果(AUC)：'income'：0.942, 'marital': 0.977  原文是0.941， 0.9927

参考论文：J.Ma et al, Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts, KDD 2018.
          Z.Zhao et al, Recommending What Video to Watch Next: A Multitask Ranking System, RecSys 2019.
          Hongyan Tang, Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations, RecSys 2020.

5 PLE 也实现了下，一起放在mmoe_model文件夹下了，使用时把mmoe.py中导入模型改为PLE就行了。实验结果与MMOE差不多。

实验结果(AUC)：'income'：0.939, 'marital': 0.979 


6 实现DeepFM，在sample Criteo（是对kaggle Criteo数据集的采样，有1000000个样本）数据上实验了一下，主要参考了这位的代码 https://blog.csdn.net/springtostring/article/details/108157070 但我觉得他的模型部分写的不太对 又自己重新写了。

实验结果(AUC): 0.743

参考论文：HGUO et al, DeepFM: a factorization-machine based neural network for CTR prediction, IJCAI 2017.

7 实现NFM, 数据集同上，主要参考DeepCTR，

实验结果(AUC): 0.710

参考论文：Xiangnan He et al, Neural Factorization Machines for Sparse Predictive Analytics, SIGIR 2017.

8 实现DCN，数据集及参考同上

实验结果(AUC): 0.750

参考论文：Ruoxi Wang et al, Deep & Cross Network for Ad Click Predictions, ADKDD 2017.

9 实现AFM, 数据集及参考同上

实验结果(AUC): 0.715

参考论文：Jun Xiao et al, Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, IJCAI 2017.

10 实现AutoInt, 数据集及参考同上

实验结果(AUC)：0.717

参考论文：Weiping Song et al, AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks, CIKM 2019.

11 实现ONN，数据集及参考同上

实验结果(AUC)：0.735

参考论文：Yi Yang et al, Operation-aware Neural Networks for user response prediction, Neural Networks 2020.

12 实现FiBiNET，数据集及参考同上

实验结果(AUC)：0.698

参考论文：Tongwen Huang et al, FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction, RecSys 2019.

13 实现Wide&Deep，数据集及参考同上

实验结果(AUC)：0.728

参考论文：Heng-Tze Cheng et al, Wide & Deep Learning for Recommender Systems, RecSys workshop/dlrs 2016.

14 实现DCN-v2，数据集及参考同上

实验结果(AUC)：0.748

参考论文：Ruoxi Wang et al, DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems, WWW 2021.

15 实现AFN，数据集及参考同上

实验结果(AUC)：0.720

参考论文：Weiyu Cheng et al, Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions, AAAI 2020.

16 实现DCAP，数据集同上

实验结果(AUC)：0.709

参考论文：Zekai Chen et al, DCAP: Deep Cross Attentional Product Network for User Response Prediction, CIKM 2021.
