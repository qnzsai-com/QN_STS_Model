# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import math
from transformers.models.bert.modeling_tf_bert import TFBertModel
from setting import ARCCSE_PRE_RANK_USE_ARC_LOSS, ARCCSE_PRE_RANK_USE_TRI_LOSS, ARCCSE_PRE_RANK_USE_DCL_LOSS


class ArcCSE(tf.keras.Model):
    def __init__(self, config, tokenizer, model_path=None):
        super(ArcCSE, self).__init__()
        self.config = config
        self.config.output_hidden_states = True
        self.tokenizer = tokenizer
        self.max_len = config.max_len
        self.use_tri_loss = ARCCSE_PRE_RANK_USE_TRI_LOSS
        self.use_arc_loss = ARCCSE_PRE_RANK_USE_ARC_LOSS
        if isinstance(config.emb_layer, int):
            self.emb_layer = [config.emb_layer]
        else:
            self.emb_layer = config.emb_layer
        if model_path:
            try:
                self.bert = TFBertModel.from_pretrained(model_path, from_pt=True, config=self.config)
            except:
                self.bert = TFBertModel.from_pretrained(model_path, from_pt=False, config=self.config)
        else:
            self.bert = TFBertModel(config=self.config)

    def call(self, inputs, **kwargs):
        input_ids = inputs["input_ids"]
        token_type_ids = inputs.get("token_type_ids", None)
        attention_mask = inputs.get("attention_mask", None)
        if self.use_tri_loss:
            n_samples = tf.shape(input_ids)[0] // 5
            emb1 = self.get_encoder_layer(input_ids=input_ids[:n_samples*2], token_type_ids=token_type_ids[:n_samples*2], attention_mask=attention_mask[:n_samples*2], **kwargs)  # [batch, dim]
            kwargs["training"] = False
            emb2 = self.get_encoder_layer(input_ids=input_ids[n_samples*2:], token_type_ids=token_type_ids[n_samples*2:], attention_mask=attention_mask[n_samples*2:], **kwargs)  # [batch, dim]
            emb = tf.concat([emb1, emb2], axis=0)
        else:
            emb = self.get_encoder_layer(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, **kwargs)  # [batch, dim]
        return emb

    def get_encoder_layer(self, input_ids, attention_mask, token_type_ids, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            **kwargs
        )
        hidden_states = outputs[2]

        if isinstance(self.emb_layer, str):
            emb = hidden_states[-1][:, 0, :]
        elif isinstance(self.emb_layer, list):
            emb = None
            for layer in self.emb_layer:
                emb = hidden_states[layer] if emb is None else emb + hidden_states[layer]
            emb = emb / len(self.emb_layer)
            emb = tf.reduce_mean(emb, axis=1)
        else:
            raise Exception(f"{self.__class__.__name__} 参数出错: emb_layer {self.emb_layer}")
        return emb

    def get_data(self, texts, max_len, return_tensor=False):
        if isinstance(texts, str):
            texts = [texts]
        data = {
            "input_ids": np.zeros(shape=(len(texts), max_len), dtype=np.int32),
            "token_type_ids": np.zeros(shape=(len(texts), max_len), dtype=np.int32),
            "attention_mask": np.zeros(shape=(len(texts), max_len), dtype=np.int32),
        }
        for i, text in enumerate(texts):
            inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_len, truncation=True, padding="max_length")
            data["input_ids"][i, :] = inputs["input_ids"]
            data["token_type_ids"][i, :] = inputs["token_type_ids"]
            data["attention_mask"][i, :] = inputs["attention_mask"]

        if return_tensor:
            data["input_ids"] = tf.convert_to_tensor(data["input_ids"])
            data["token_type_ids"] = tf.convert_to_tensor(data["token_type_ids"])
            data["attention_mask"] = tf.convert_to_tensor(data["attention_mask"])
        return data

class ArcLoss2N(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(ArcLoss2N, self).__init__(**kwargs)
        self.use_tri_loss = ARCCSE_PRE_RANK_USE_TRI_LOSS
        self.use_arc_loss = ARCCSE_PRE_RANK_USE_ARC_LOSS
        self.use_dcl_loss = ARCCSE_PRE_RANK_USE_DCL_LOSS
        self.temperature = 20
        self.margin_angle = math.pi / 18
        self.margin_tri_loss = 0.0
        self.lambda_ratio = 0.1

        # 增加角度 margin 需要的参数
        self.margin_angle = math.pi / 18
        self.cos_m = math.cos(self.margin_angle)
        self.sin_m = math.sin(self.margin_angle)
        self.th = math.cos(math.pi - self.margin_angle)
        self.mm = math.sin(math.pi - self.margin_angle) * self.margin_angle
        self.easy_margin = True

    def call(self, y_true, y_pred):
        """
        一个 batch 的句子 [b1, b2, b3, b4, ...], 每两个是语义相同的 b1 和 b2 相同, b3 和 b4 相同
        y_true: 无用全0, 要在call中重新计算,计算得到 [batch, batch] 的矩阵
        y_pred: [batch, dim] 每个句子的 embedding
        """
        # 构造标签
        n_samples = tf.shape(y_pred)[0] // 5 if self.use_tri_loss else tf.shape(y_pred)[0] // 2
        idxs = tf.range(0, n_samples*2)
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        y_true = tf.equal(idxs_1, idxs_2)
        y_true = tf.cast(y_true, tf.float32)

        # 计算相似度
        y_pred = tf.math.l2_normalize(y_pred, axis=1)
        y_pred_pos = y_pred[:n_samples*2] if self.use_tri_loss else y_pred
        y_pred_tri_0 = y_pred[n_samples*2:][0::3] if self.use_tri_loss else None
        y_pred_tri_1 = y_pred[n_samples*2:][1::3] if self.use_tri_loss else None
        y_pred_tri_2 = y_pred[n_samples*2:][2::3] if self.use_tri_loss else None

        # 计算 pair_loss
        if self.use_arc_loss:
            cosine = tf.matmul(y_pred_pos, y_pred_pos, transpose_b=True)
            cosine = cosine - tf.eye(n_samples*2) * 1e12
            cosine = tf.clip_by_value(cosine, clip_value_min=-1, clip_value_max=1)
            sine = tf.sqrt(tf.clip_by_value((1.0 - tf.pow(cosine, 2)), clip_value_min=0, clip_value_max=1))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = tf.where(cosine > 0, phi, cosine)
            else:
                phi = tf.where(cosine > self.th, phi, cosine - self.mm)
            temp_eye = tf.eye(n_samples*2)
            similarities = cosine * (1 - temp_eye) + phi * temp_eye
        else:
            similarities = tf.matmul(y_pred_pos, y_pred_pos, transpose_b=True)
            similarities = similarities - tf.eye(n_samples*2) * 1e12
        similarities = similarities * self.temperature

        if self.use_dcl_loss:
            similarities_exp = tf.exp(similarities)
            similarities_pos = tf.reduce_sum(similarities_exp * y_true, -1)
            similarities_denom = tf.reduce_sum(similarities_exp * (1-y_true), -1)
            loss = -1 * tf.math.log(similarities_pos/similarities_denom)
            loss = tf.reduce_mean(loss)
        else:
            loss = tf.keras.losses.categorical_crossentropy(y_true, similarities, from_logits=True)
            loss = tf.reduce_mean(loss)

        # 计算 tri_loss
        if self.use_tri_loss:
            similarities_tri_1 = tf.matmul(y_pred_tri_0, y_pred_tri_1, transpose_b=True)
            similarities_tri_2 = tf.matmul(y_pred_tri_0, y_pred_tri_2, transpose_b=True)
            loss_tri = tf.maximum(0.0, similarities_tri_2 - similarities_tri_1 + self.margin_tri_loss)
            loss_tri = tf.reduce_mean(loss_tri)
            loss = loss + self.lambda_ratio * loss_tri

        return loss

    def get_config(self):
        return super(ArcLoss2N, self).get_config()

class ArcLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(ArcLoss, self).__init__(**kwargs)
        self.use_tri_loss = ARCCSE_PRE_RANK_USE_TRI_LOSS
        self.use_arc_loss = ARCCSE_PRE_RANK_USE_ARC_LOSS
        self.use_dcl_loss = ARCCSE_PRE_RANK_USE_DCL_LOSS
        self.temperature = 20
        self.margin_angle = math.pi / 18
        self.margin_tri_loss = 0.0
        self.lambda_ratio = 0.1

        # 增加角度 margin 需要的参数
        self.margin_angle = math.pi / 18
        self.cos_m = math.cos(self.margin_angle)
        self.sin_m = math.sin(self.margin_angle)
        self.th = math.cos(math.pi - self.margin_angle)
        self.mm = math.sin(math.pi - self.margin_angle) * self.margin_angle
        self.easy_margin = True

    def call(self, y_true, y_pred):
        """
        一个 batch (2n个) 的句子 [a1, a2, a3, a4, ..., an, b1, b2, b3, b4, ..., bn], ai 和 bi 是相同语义的
        y_true: 无用全0, 在call中重新计算, [0, 1, 2, 3, ..., n-1]
        y_pred: [2n, dim] 每个句子的 embedding
        """
        # 构造标签
        n_samples = tf.shape(y_pred)[0] // 5 if self.use_tri_loss else tf.shape(y_pred)[0] // 2
        y_true = tf.range(0, n_samples)
        y_pred = tf.math.l2_normalize(y_pred, axis=1)
        y_pred_pos_1 = y_pred[:n_samples*2][0::2] if self.use_tri_loss else y_pred[0::2]
        y_pred_pos_2 = y_pred[:n_samples*2][1::2] if self.use_tri_loss else y_pred[1::2]
        y_pred_tri_0 = y_pred[n_samples*2:][0::3] if self.use_tri_loss else None
        y_pred_tri_1 = y_pred[n_samples*2:][1::3] if self.use_tri_loss else None
        y_pred_tri_2 = y_pred[n_samples*2:][2::3] if self.use_tri_loss else None

        # 计算 pair_loss
        if self.use_arc_loss:
            cosine = tf.matmul(y_pred_pos_1, y_pred_pos_2, transpose_b=True)
            sine = tf.sqrt(tf.clip_by_value((1.0-tf.pow(cosine, 2)), clip_value_min=0, clip_value_max=1))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = tf.where(cosine > 0, phi, cosine)
            else:
                phi = tf.where(cosine > self.th, phi, cosine - self.mm)
            temp_eye = tf.eye(n_samples)
            similarities = cosine*(1-temp_eye)+phi*temp_eye
        else:
            similarities = tf.matmul(y_pred_pos_1, y_pred_pos_2, transpose_b=True)
        similarities = similarities * self.temperature

        if self.use_dcl_loss:
            similarities_exp = tf.exp(similarities)
            diag_matrix = tf.eye(tf.shape(similarities_exp)[0])
            similarities_pos = tf.reduce_sum(similarities_exp * diag_matrix, -1)
            similarities_denom = tf.reduce_sum(similarities_exp * (1-diag_matrix), -1)
            loss = -1 * tf.math.log(similarities_pos/similarities_denom)
            loss = tf.reduce_mean(loss)
        else:
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, similarities, from_logits=True)
            loss = tf.reduce_mean(loss)

        # 计算 tri_loss
        if self.use_tri_loss:
            similarities_tri_1 = tf.matmul(y_pred_tri_0, y_pred_tri_1, transpose_b=True)
            similarities_tri_2 = tf.matmul(y_pred_tri_0, y_pred_tri_2, transpose_b=True)
            loss_tri = tf.maximum(0.0, similarities_tri_2-similarities_tri_1+self.margin_tri_loss)
            loss_tri = tf.reduce_mean(loss_tri)
            loss = loss + self.lambda_ratio * loss_tri

        return loss

    def get_config(self):
        return super(ArcLoss, self).get_config()

class ArcMetric2N(tf.keras.metrics.Metric):
    def __init__(self, metric_type="acc", name='sim_acc', threshold=0.7, **kwargs):
        super(ArcMetric2N, self).__init__(name=name, **kwargs)
        self.metric_type = metric_type
        self.threshold = threshold
        self.tp = tf.metrics.TruePositives(thresholds=self.threshold)
        self.fp = tf.metrics.FalsePositives(thresholds=self.threshold)
        self.tn = tf.metrics.TrueNegatives(thresholds=self.threshold)
        self.fn = tf.metrics.FalseNegatives(thresholds=self.threshold)
        self.auc = tf.metrics.AUC()
        self.use_tri_loss = ARCCSE_PRE_RANK_USE_TRI_LOSS
        self.use_arc_loss = ARCCSE_PRE_RANK_USE_ARC_LOSS

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 构造标签
        n_samples = tf.shape(y_pred)[0] // 5 if self.use_tri_loss else tf.shape(y_pred)[0] // 2
        idxs = tf.range(0, n_samples*2)
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        y_true = tf.equal(idxs_1, idxs_2)
        y_true = tf.cast(y_true, tf.int32)

        # 计算相似度
        y_pred = tf.math.l2_normalize(y_pred, axis=1)
        y_pred_pos = y_pred[:n_samples*2] if self.use_tri_loss else y_pred
        similarities = tf.matmul(y_pred_pos, y_pred_pos, transpose_b=True)
        similarities = similarities - tf.eye(n_samples*2) * 2
        y_pred = tf.clip_by_value((similarities+1)/2, clip_value_min=0, clip_value_max=1)

        self.auc.update_state(y_true=y_true, y_pred=y_pred)
        self.tp.update_state(y_true=y_true, y_pred=y_pred)
        self.fp.update_state(y_true=y_true, y_pred=y_pred)
        self.tn.update_state(y_true=y_true, y_pred=y_pred)
        self.fn.update_state(y_true=y_true, y_pred=y_pred)

    def result(self):
        tp = self.tp.result()
        fp = self.fp.result()
        tn = self.tn.result()
        fn = self.fn.result()
        auc = self.auc.result()

        if self.metric_type == "auc":
            return auc
        elif self.metric_type == "acc":
            return (tp+tn)/(tp+fp+tn+fn)
        elif self.metric_type == "precision":
            return tp/(tp+fp)
        elif self.metric_type == "recall":
            return tp/(tp+fn)
        else:
            p = tp/(tp+fp)
            r = tp/(tp+fn)
            return 2*p*r/(p+r)

    def reset_states(self):
        self.tp.reset_states()
        self.fp.reset_states()
        self.tn.reset_states()
        self.fn.reset_states()
        self.auc.reset_states()

class ArcMetric(tf.keras.metrics.Metric):
    def __init__(self, metric_type="acc", name='sim_acc', threshold=0.7, **kwargs):
        super(ArcMetric, self).__init__(name=name, **kwargs)
        self.metric_type = metric_type
        self.threshold = threshold
        self.tp = tf.metrics.TruePositives(thresholds=self.threshold)
        self.fp = tf.metrics.FalsePositives(thresholds=self.threshold)
        self.tn = tf.metrics.TrueNegatives(thresholds=self.threshold)
        self.fn = tf.metrics.FalseNegatives(thresholds=self.threshold)
        self.auc = tf.metrics.AUC()
        self.use_tri_loss = ARCCSE_PRE_RANK_USE_TRI_LOSS
        self.use_arc_loss = ARCCSE_PRE_RANK_USE_ARC_LOSS

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 构造标签
        n_samples = tf.shape(y_pred)[0] // 5 if self.use_tri_loss else tf.shape(y_pred)[0] // 2
        y_true = tf.eye(n_samples)

        # 计算相似度
        y_pred = tf.math.l2_normalize(y_pred, axis=1)
        y_pred_pos_1 = y_pred[:n_samples*2][0::2] if self.use_tri_loss else y_pred[0::2]
        y_pred_pos_2 = y_pred[:n_samples*2][1::2] if self.use_tri_loss else y_pred[1::2]
        similarities = tf.matmul(y_pred_pos_1, y_pred_pos_2, transpose_b=True)
        y_pred = tf.clip_by_value((similarities+1)/2, clip_value_min=0, clip_value_max=1)

        self.auc.update_state(y_true=y_true, y_pred=y_pred)
        self.tp.update_state(y_true=y_true, y_pred=y_pred)
        self.fp.update_state(y_true=y_true, y_pred=y_pred)
        self.tn.update_state(y_true=y_true, y_pred=y_pred)
        self.fn.update_state(y_true=y_true, y_pred=y_pred)

    def result(self):
        tp = self.tp.result()
        fp = self.fp.result()
        tn = self.tn.result()
        fn = self.fn.result()
        auc = self.auc.result()

        if self.metric_type == "auc":
            return auc
        elif self.metric_type == "acc":
            return (tp+tn)/(tp+fp+tn+fn)
        elif self.metric_type == "precision":
            return tp/(tp+fp)
        elif self.metric_type == "recall":
            return tp/(tp+fn)
        else:
            p = tp/(tp+fp)
            r = tp/(tp+fn)
            return 2*p*r/(p+r)

    def reset_states(self):
        self.tp.reset_states()
        self.fp.reset_states()
        self.tn.reset_states()
        self.fn.reset_states()
        self.auc.reset_states()
