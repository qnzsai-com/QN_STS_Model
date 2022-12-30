#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:admin
@file:RUN_ZERO_CLUE.py
@time:2022/12/30
"""

from setting import ARCCSE_PRE_RANK_EMB_LAYER
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from transformers import BertConfig, BertTokenizer
from ArcCSE import ArcCSE
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


PRETRAIN_BERT_NAME = 'arccse_pretrain_base'
PRETRAIN_BERT_DIR = f'./pretrain_model/{PRETRAIN_BERT_NAME}'
PRETRAIN_BERT_CONFIG = f'./pretrain_model/{PRETRAIN_BERT_NAME}/config.json'
PRETRAIN_BERT_VOCAB = f'./pretrain_model/{PRETRAIN_BERT_NAME}/vocab.txt'
PRETRAIN_BERT_MODEL_PT = f'./pretrain_model/{PRETRAIN_BERT_NAME}/pytorch_model.bin'
PRETRAIN_BERT_MODEL_TF = f'./pretrain_model/{PRETRAIN_BERT_NAME}/tf_model.h5'

# 加载预训练模型
config = BertConfig.from_pretrained(PRETRAIN_BERT_DIR)
config.emb_layer = ARCCSE_PRE_RANK_EMB_LAYER
config.max_len = 128
tokenizer = BertTokenizer.from_pretrained(PRETRAIN_BERT_DIR)
model = ArcCSE(config=config, tokenizer=tokenizer, model_path=PRETRAIN_BERT_DIR)

# 加载数据
valid_data_list = []
with open("/data/cbk/aicc-nlu/data/CLUE/ZeroCLUE/datasets/bustm/dev.json", "r", encoding="utf-8") as f:
    for line in f:
        valid_data_list.append(json.loads(line.strip()))
# 验证
valid_scores_list = []
for i, data in tqdm(enumerate(valid_data_list)):
    texts = [data["sentence1"], data["sentence2"]]
    inputs = model.get_data(texts=texts, max_len=config.max_len, return_tensor=True)
    embeddings = model.get_encoder_layer(**inputs)
    embeddings = embeddings.numpy()
    scores = cosine_similarity(embeddings, embeddings)
    data["score"] = scores[0, 1]
    valid_scores_list.append(scores[0, 1])

best_acc = 0
best_threshold = 0
for t in range(30, 100, 1):
    threshold = t/100.0
    acc = 0
    for d in valid_data_list:
        if d["label"] == "1" and d["score"] > threshold:
            acc += 1
        if d["label"] == "0" and d["score"] <= threshold:
            acc += 1
    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold
print(f"best_threshold:{best_threshold}, best_acc:{best_acc/len(valid_data_list)}")

# 加载数据
data_list = []
with open("/data/cbk/aicc-nlu/data/CLUE/ZeroCLUE/datasets/bustm/test.json", "r", encoding="utf-8") as f:
    for line in f:
        data_list.append(json.loads(line.strip()))
# 预测
scores_list = []
for i, data in tqdm(enumerate(data_list)):
    texts = [data["sentence1"], data["sentence2"]]
    inputs = model.get_data(texts=texts, max_len=config.max_len, return_tensor=True)
    embeddings = model.get_encoder_layer(**inputs)
    embeddings = embeddings.numpy()
    scores = cosine_similarity(embeddings, embeddings)
    data["score"] = scores[0, 1]
    scores_list.append(scores[0, 1])

# 画图
from matplotlib import pyplot as plt
scores_list.sort()
plt.figure()
plt.title(f"scores")
plt.hist(scores_list, bins=100)
plt.xlim(0.0, 1)
plt.savefig(f"./clue_scores.jpg")
plt.show()

# 保存结果
threshold = 0.68
with open("./bustm_predict.json", "w", encoding="utf-8") as f:
    for d in data_list:
        result = {"id": d["id"], "label": "1" if d["score"] > threshold else "0"}
        f.writelines(f"{json.dumps(result)}\n")
print("wait")
