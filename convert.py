#! -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


keep_weights = [
    'bert/embeddings/position_embeddings',
    'bert/embeddings/token_type_embeddings',
    'bert/embeddings/LayerNorm/gamma',
    'bert/embeddings/LayerNorm/beta',
    'bert/pooler/dense/kernel',
    'bert/pooler/dense/bias',
    'cls/seq_relationship/output_weights',
    'cls/seq_relationship/output_bias',
    'cls/predictions/transform/dense/kernel',
    'cls/predictions/transform/dense/bias',
    'cls/predictions/transform/LayerNorm/gamma',
    'cls/predictions/transform/LayerNorm/beta',
    'cls/predictions/output_bias',
]

albert_block = [
    'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel',
    'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias',
    'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel',
    'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias',
    'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel',
    'bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias',
    'bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel',
    'bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias',
    'bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma',
    'bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta',
    'bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel',
    'bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias',
    'bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel',
    'bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias',
    'bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma',
    'bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta',
]

albert_block_brightmart = [
    'bert/encoder/layer_shared/attention/self/query/kernel',
    'bert/encoder/layer_shared/attention/self/query/bias',
    'bert/encoder/layer_shared/attention/self/key/kernel',
    'bert/encoder/layer_shared/attention/self/key/bias',
    'bert/encoder/layer_shared/attention/self/value/kernel',
    'bert/encoder/layer_shared/attention/self/value/bias',
    'bert/encoder/layer_shared/attention/output/dense/kernel',
    'bert/encoder/layer_shared/attention/output/dense/bias',
    'bert/encoder/layer_shared/attention/output/LayerNorm/gamma',
    'bert/encoder/layer_shared/attention/output/LayerNorm/beta',
    'bert/encoder/layer_shared/intermediate/dense/kernel',
    'bert/encoder/layer_shared/intermediate/dense/bias',
    'bert/encoder/layer_shared/output/dense/kernel',
    'bert/encoder/layer_shared/output/dense/bias',
    'bert/encoder/layer_shared/output/LayerNorm/gamma',
    'bert/encoder/layer_shared/output/LayerNorm/beta',
]


checkpoint_file = 'albert_base_zh_additional_36k_steps/albert_model.ckpt'


def load_variable(name):
    variable = tf.train.load_variable(checkpoint_file, name)
    return variable


def create_variable(name, value):
    return tf.Variable(value, name=name)


with tf.Graph().as_default():

    weight_1 = load_variable('bert/embeddings/word_embeddings')
    weight_2 = load_variable('bert/embeddings/word_embeddings_2')
    weight = np.dot(weight_1, weight_2)
    create_variable('bert/embeddings/word_embeddings', weight)

    for name in keep_weights:
        weight = load_variable(name)
        create_variable(name, weight)

    for name_1, name_2 in zip(albert_block_brightmart, albert_block):
        weight = load_variable(name_1)
        create_variable(name_2, weight)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, 'albert_base_google_zh_additional_36k_steps/albert_model.ckpt', write_meta_graph=False)
