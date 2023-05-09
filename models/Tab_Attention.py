from collections import OrderedDict
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Input, GlobalAveragePooling1D
from tensorflow.keras.layers import Input, Dense, concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Layer

# torch.backends.cudnn.benchmark = True
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class Self_Attention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim):
        super(Self_Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.query_dense = tf.keras.layers.Dense(embedding_dim)
        self.key_dense = tf.keras.layers.Dense(embedding_dim)
        self.value_dense = tf.keras.layers.Dense(embedding_dim)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        # Split inputs into queries, keys, and values
        queries = self.query_dense(inputs)
        keys = self.key_dense(inputs)
        values = self.value_dense(inputs)

        # Calculate attention weights and weighted sum of values
        attention_logits = tf.matmul(queries, keys, transpose_b=True)
        attention_weights = self.softmax(attention_logits)
        attention_output = tf.matmul(attention_weights, values)

        return attention_output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embedding_dim': self.embedding_dim,
            'query_dense': self.query_dense,
            'key_dense': self.key_dense,
            'value_dense': self.value_dense,
            'softmax': self.softmax
        })
        return config


def build_gnns(output_shape, groups_num, data_name, attention=False):

    inputs = [Input(shape=shape) for shape in groups_num]

    results = []
    l = 0
    activation = 'relu'
    if 'lc' in data_name:
        l_num = [[60, 20, 5], [40, 20, 10]]  # 19
        # l_num = [[60, 20], [40, 20]]
    else:
        l_num = [[20, 5], [20, 10]]

    if data_name == 'lc':
        all_output = Dense(l_num[0][0], activation=activation)(inputs[0])
        all_output = BatchNormalization()(all_output)
        for l_n in l_num[0][1:-1]:
            all_output = Dense(l_n, activation=activation)(all_output)
    else:
        for l_n in l_num[0][:-1]:
            all_output = Dense(l_n, activation=activation)(inputs[0])
            all_output = BatchNormalization()(all_output)

        all_output = Dense(l_num[0][-1], activation=activation)(all_output)

    for i in range(1, len(groups_num)):
        # define group input
        group_input = inputs[i]
        # obtain group output
        if data_name == 'lc':
            group_output = Dense(l_num[1][0], activation=activation)(group_input)
            group_output = BatchNormalization()(group_output)
            for l_n in l_num[1][1:-1]:
                group_output = Dense(l_n, activation=activation)(group_output)
        else:
            for l_n in l_num[1][:-1]:
                group_output = Dense(l_n, activation=activation)(group_input)
                group_output = BatchNormalization()(group_output)
            group_output = Dense(l_num[1][-1], activation=activation)(group_output)
        if 'lc' in data_name:# and data_name != 'lc2019':
            group_output = BatchNormalization()(group_output)
        results.append(Dense(1, activation='sigmoid', name='output' + str(l))(group_output))
        l += 1


    merged = concatenate([all_output] + results)
    if 'lc' in data_name and data_name != 'lc2019':
        merged = BatchNormalization()(merged)

    if attention == True:
        output = Self_Attention(15)(merged)
    else:
        output = Dense(15, activation=activation)(merged)

    output = Dense(output_shape, activation='sigmoid', name='output')(output)

    # define model
    model = Model(inputs=inputs, outputs=[output] + results)

    return model


def dnn_model(output_shape, input_shape, hidden_layer=[80, 40]):
    # 初始化输入层
    inputs = Input(shape=input_shape)
    # inputs = RBFLayer(16, 0.5)(inputs)
    y0 = Dense(hidden_layer[0], activation='relu')(inputs)
    y1 = BatchNormalization()(y0)
    # y1 = Dropout(dropout_rate)(y1)
    y2 = Dense(hidden_layer[1], activation='relu')(y1)
    y2 = BatchNormalization()(y2)
    # y2 = Dense(hidden_layer[2], activation='relu')(y2)
    # y2 = BatchNormalization()(y2)
    # y2 = Dense(hidden_layer[3], activation='relu')(y2)
    # y2 = BatchNormalization()(y2)
    # y2 = Dense(hidden_layer[4], activation='relu')(y2)
    # 定义输出层
    output = Dense(output_shape, activation='sigmoid')(y2)
    # 定义模型
    model = Model(inputs=inputs, outputs=output)
    return model


def build_dnn(output_shape, input_shape, hidden_layer=[], dropout_rate=0.2):
    # 初始化输入层
    inputs = Input(shape=input_shape)
    y0 = Dense(hidden_layer[0], activation='relu')(inputs)
    y1 = BatchNormalization()(y0)
    # y1 = Dropout(dropout_rate)(y1)
    y2 = Dense(hidden_layer[1], activation='relu')(y1)
    y2 = BatchNormalization()(y2)
    y2 = Dense(hidden_layer[2], activation='relu')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Dense(hidden_layer[3], activation='relu')(y2)
    y2 = BatchNormalization()(y2)
    y2 = Dense(hidden_layer[4], activation='relu')(y2)
    # 定义输出层
    output = Dense(output_shape, activation='sigmoid')(y2)
    # 定义模型
    model = Model(inputs=inputs, outputs=output)
    return model


class dnn(tf.keras.Model):
    def __init__(self):
        super(dnn, self).__init__(name='')

        self.dense1 = tf.keras.layers.Dense(80, activation='relu')
        self.dense2 = tf.keras.layers.Dense(40, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, input_tensor):
        x = self.dense1(input_tensor)
        x = self.dense2(x)
        output = self.dense3(x)
        return output

class F1_Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = tf.keras.metrics.Precision(thresholds=0.5)
        self.recall_fn = tf.keras.metrics.Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        return self.f1

    def reset_states(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)
