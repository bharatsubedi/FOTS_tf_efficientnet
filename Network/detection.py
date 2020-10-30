import collections

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from utils import config
from Network.base_model import efficientnet_builder


def un_pool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])


def mean_image_subtraction(images):
    means = [116.78]
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


class Detection(object):
    def __init__(self, is_training=True):
        self.is_training = is_training

    def build_graph(self, images, weight_decay=1e-5):
        images = mean_image_subtraction(images)
        #select efficientnet version from b0 to b7
        _, end_points = efficientnet_builder.build_model(images, 'efficientnet-b4', self.is_training)
        # print(images)

        with tf.variable_scope('feature_fusion', values=[end_points.values]):
            batch_norm_params = {'decay': 0.997, 'epsilon': 1e-5, 'scale': True, 'is_training': self.is_training}
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(weight_decay)):
                f = [end_points['reduction_5'], end_points['reduction_4'], end_points['reduction_3'], end_points['reduction_2']]
                g = [None, None, None, None]
                h = [None, None, None, None]
                num_outputs = [None, 128, 64, 32]
                for i in range(4):
                    if i == 0:
                        h[i] = f[i]
                    else:
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], f[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= 2:
                        g[i] = un_pool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                f_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * config.text_scale
                angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi / 2
                f_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return g[3], f_score, f_geometry

    @staticmethod
    def dice_coefficient(y_true_cls, y_pred_cls, training_mask):
        eps = 1e-5
        intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
        union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
        loss = 1. - (2 * intersection / union)
        tf.compat.v1.summary.scalar('classification_dice_loss', loss)
        return loss

    def loss(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):
        classification_loss = self.dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        classification_loss *= 0.01
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        l_aabb = -tf.math.log((area_intersect + 1.0) / (area_union + 1.0))
        l_theta = 1 - tf.cos(theta_pred - theta_gt)
        tf.compat.v1.summary.scalar('geometry_AABB', tf.reduce_mean(l_aabb * y_true_cls * training_mask))
        tf.compat.v1.summary.scalar('geometry_theta', tf.reduce_mean(l_theta * y_true_cls * training_mask))
        l_g = l_aabb + 20 * l_theta
        return tf.reduce_mean(l_g * y_true_cls * training_mask) + classification_loss
