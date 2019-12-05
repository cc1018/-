# coding: utf-8

from __future__ import print_function

import jieba
import os
import tensorflow as tf
import tensorflow.contrib.keras as kr
import numpy as np
import re
from cnn_model_1 import TCNNConfig, TextCNN
from process_data import read_category, getWordEmbedding, read_vocab

try:
    bool(type(unicode))
except NameError:
    unicode = str
# fea_re = re.compile(u'[a-zA-Z0-9\u4e00-\u9fa5]+')
# http_re = re.compile(r'http://[a-zA-Z0-9\.\/]*')
# emotion_re = re.compile(r'\[(.*?)\]')
base_dir = 'data'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.ci')

save_dir = 'model/ci-128-False'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def native_content(content):
    return content

def open_file(filename, mode='r'):
        return open(filename, mode, encoding='utf-8', errors='ignore')

def read_file(filename):
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(native_content(content))
                    # contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels

class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.word_to_id, wordEmbedding = getWordEmbedding(vocab_dir)
        # self.words, self.word_to_id = read_vocab(vocab_dir)
        # self.config.vocab_size = len(self.word_to_id)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        # data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        data = [self.word_to_id[x] for x in content.split(" ") if x in self.word_to_id]
        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }
        y_pred = self.session.run(self.model.y_pred, feed_dict=feed_dict)
        y_pred_cls = np.argmax(y_pred,1)

        return self.categories[y_pred_cls[0]], str(np.max(y_pred))

if __name__ == '__main__':
    cnn_model = CnnModel()
    # test_demo = ['马晓旭 意外 受伤 让 国奥 警惕 无奈 大雨 格外 青睐 殷家 军 记者 傅亚雨 沈阳 报道 来到 沈阳']
    # cnn_model.predict(test_demo)
    test_dir = './data/cnews.test.ci'
    contents, labels = read_file(test_dir)
    fd = open('result.txt','w')
    for i in range(len(contents)):
        pre, logit = cnn_model.predict(contents[i])
        fd.write(labels[i]+'\t'+pre+'\t'+logit+'\n')