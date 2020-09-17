from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from utils import config


def un_pool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])


def mean_image_subtraction(images):
    means = [123.68, 116.78, 103.94]
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

MOBILE_NET_V1_CONV_DEFS = [Conv(kernel=[3, 3], stride=2, depth=32),
                           DepthSepConv(kernel=[3, 3], stride=1, depth=64),
                           DepthSepConv(kernel=[3, 3], stride=2, depth=128),
                           DepthSepConv(kernel=[3, 3], stride=1, depth=128),
                           DepthSepConv(kernel=[3, 3], stride=2, depth=256),
                           DepthSepConv(kernel=[3, 3], stride=1, depth=256),
                           DepthSepConv(kernel=[3, 3], stride=2, depth=512),
                           DepthSepConv(kernel=[3, 3], stride=1, depth=512),
                           DepthSepConv(kernel=[3, 3], stride=1, depth=512),
                           DepthSepConv(kernel=[3, 3], stride=1, depth=512),
                           DepthSepConv(kernel=[3, 3], stride=1, depth=512),
                           DepthSepConv(kernel=[3, 3], stride=1, depth=512),
                           DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
                           DepthSepConv(kernel=[3, 3], stride=1, depth=1024)]


def _fixed_padding(inputs, kernel_size, rate=1):
    kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                             kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]], [pad_beg[1], pad_end[1]], [0, 0]])
    return padded_inputs


def mobile_net_v1_base(inputs,
                       final_endpoint='Conv2d_13_pointwise',
                       min_depth=8,
                       depth_multiplier=1.0,
                       conv_defs=None,
                       output_stride=None,
                       use_explicit_padding=False,
                       scope=None):
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if conv_defs is None:
        conv_defs = MOBILE_NET_V1_CONV_DEFS

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    padding = 'SAME'
    if use_explicit_padding:
        padding = 'VALID'
    with tf.variable_scope(scope, 'MobileNetV1', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding=padding):
            current_stride = 1
            rate = 1
            net = inputs
            for i, conv_def in enumerate(conv_defs):
                end_point_base = 'Conv2d_%d' % i
                if output_stride is not None and current_stride == output_stride:
                    layer_stride = 1
                    layer_rate = rate
                    rate *= conv_def.stride
                else:
                    layer_stride = conv_def.stride
                    layer_rate = 1
                    current_stride *= conv_def.stride
                if isinstance(conv_def, Conv):
                    end_point = end_point_base
                    if use_explicit_padding:
                        net = _fixed_padding(net, conv_def.kernel)
                    net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel, stride=conv_def.stride,
                                      scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

                elif isinstance(conv_def, DepthSepConv):
                    end_point = end_point_base + '_depthwise'
                    if use_explicit_padding:
                        net = _fixed_padding(net, conv_def.kernel, layer_rate)
                    net = slim.separable_conv2d(net, None, conv_def.kernel,
                                                depth_multiplier=1,
                                                stride=layer_stride,
                                                rate=layer_rate,
                                                scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                    end_point = end_point_base + '_pointwise'
                    net = slim.conv2d(net, depth(conv_def.depth), [1, 1], stride=1, scope=end_point)
                    end_points[end_point] = net
                else:
                    raise ValueError('Unknown convolution type %s for layer %d' % (conv_def.ltype, i))
                if i == 0:
                    end_points['pool2'] = net
                if i == 5:
                    end_points['pool3'] = net
                if i == 11:
                    end_points['pool4'] = net
                if i == 13:
                    end_points['pool5'] = net
                if end_point == final_endpoint:
                    return net, end_points                   
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def mobile_net_v1(inputs,
                  is_training=True,
                  min_depth=8,
                  depth_multiplier=1.0,
                  conv_defs=None,
                  reuse=None,
                  scope='MobileNetV1'):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))

    with tf.variable_scope(scope, 'MobileNetV1', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = mobile_net_v1_base(inputs, scope=scope,
                                                 min_depth=min_depth,
                                                 depth_multiplier=depth_multiplier,
                                                 conv_defs=conv_defs)
    return net, end_points


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                           min(shape[2], kernel_size[1])]
    return kernel_size_out


def mobile_net_v1_arg_scope(is_training=True,
                            weight_decay=0.00004,
                            stddev=0.09,
                            regularize_depth_wise=False,
                            batch_norm_decay=0.9997,
                            batch_norm_epsilon=0.001,
                            batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
                            normalizer_fn=slim.batch_norm):
    batch_norm_params = {'center': True,
                         'scale': True,
                         'decay': batch_norm_decay,
                         'epsilon': batch_norm_epsilon,
                         'updates_collections': batch_norm_updates_collections, }
    if is_training is not None:
        batch_norm_params['is_training'] = is_training
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depth_wise:
        depth_wise_regularizer = regularizer
    else:
        depth_wise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_init,
                        activation_fn=tf.nn.relu6, normalizer_fn=normalizer_fn):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d], weights_regularizer=depth_wise_regularizer) as sc:
                    return sc


class Detection(object):
    def __init__(self, is_training=True):
        self.is_training = is_training

    def build_graph(self, images, weight_decay=1e-5):
        images = mean_image_subtraction(images)
        with slim.arg_scope(mobile_net_v1_arg_scope()):
            logits, end_points = mobile_net_v1(images, is_training=self.is_training)

        with tf.variable_scope('feature_fusion', values=[end_points.values]):
            batch_norm_params = {'decay': 0.997, 'epsilon': 1e-5, 'scale': True, 'is_training': self.is_training}
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(weight_decay)):
                f = [end_points['pool5'], end_points['pool4'], end_points['pool3'], end_points['pool2']]
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
        tf.summary.scalar('classification_dice_loss', loss)
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
        l_aabb = -tf.log((area_intersect + 1.0) / (area_union + 1.0))
        l_theta = 1 - tf.cos(theta_pred - theta_gt)
        tf.summary.scalar('geometry_AABB', tf.reduce_mean(l_aabb * y_true_cls * training_mask))
        tf.summary.scalar('geometry_theta', tf.reduce_mean(l_theta * y_true_cls * training_mask))
        l_g = l_aabb + 20 * l_theta
        return tf.reduce_mean(l_g * y_true_cls * training_mask) + classification_loss
