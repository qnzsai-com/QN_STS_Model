# QN_STS_Model

预训练的语义匹配模型，采用对比学习进行预训练，优化方法包括ArcCSE、ESimCSE、DCL loss等方式，预测的时候根据embedding的余弦相似度判断句子语义是否相似。

预训练使用的数据均为STS语义匹配数据，包括：PKU-Paraphrase-Bank-master、afqmc_public、bq dataset、Icqmc、simclue_public.

模型保存在百度云，下载链接如下，下载完后把模型放在pretrain_model/arccse_pretrain_base下面就可以了：
链接：https://pan.baidu.com/s/1Ylk3UwzDtBHJpMsXD_gWlA 
提取码：4vgj