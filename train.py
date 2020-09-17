import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from utils import dataset, rotate, config
from Network import detection, recognition

detect_part = detection.Detection(is_training=True)
roi_rotate_part = rotate.RoIRotate()
recognize_part = recognition.Recognition(is_training=True)


def get_parameter():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def build_graph(input_images, input_transform_matrix, input_box_masks, input_box_widths):
    shared_feature, f_score, f_geometry = detect_part.build_graph(input_images)
    pad_rois = roi_rotate_part.roi_rotate_tensor_pad(shared_feature, input_transform_matrix, input_box_masks,
                                                     input_box_widths)
    recognition_logits, _ = recognize_part.model(pad_rois, input_box_widths)
    return f_score, f_geometry, recognition_logits


def compute_loss(f_score, f_geometry, recognition_logits, input_score_maps, input_geo_maps, input_training_masks,
                 input_transcription, input_box_widths, lamda=0.01):
    detection_loss = detect_part.loss(input_score_maps, f_score, input_geo_maps, f_geometry, input_training_masks)
    recognition_loss = recognize_part.loss(recognition_logits, input_transcription, input_box_widths)

    tf.compat.v1.summary.scalar('detect_loss', detection_loss)
    tf.compat.v1.summary.scalar('recognize_loss', recognition_loss)

    return detection_loss, recognition_loss, detection_loss + lamda * recognition_loss


def main():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_list
    if not tf.gfile.Exists(config.checkpoint_path):
        tf.gfile.MkDir(config.checkpoint_path)
    else:
        if not config.restore:
            tf.gfile.DeleteRecursively(config.checkpoint_path)
            tf.gfile.MkDir(config.checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
    input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')
    input_transcription = tf.sparse_placeholder(tf.int32, name='input_transcription')

    input_transform_matrix = tf.placeholder(tf.float32, shape=[None, 6], name='input_transform_matrix')
    input_transform_matrix = tf.stop_gradient(input_transform_matrix)
    input_box_masks = []
    input_box_widths = tf.placeholder(tf.int32, shape=[None], name='input_box_widths')
    for i in range(config.batch_size_per_gpu):
        input_box_masks.append(tf.placeholder(tf.int32, shape=[None], name='input_box_masks_' + str(i)))

    f_score, f_geometry, recognition_logits = build_graph(input_images,
                                                          input_transform_matrix,
                                                          input_box_masks,
                                                          input_box_widths)

    global_step = tf.compat.v1.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(config.learning_rate, global_step, decay_steps=10000, decay_rate=0.94,
                                               staircase=True)
    # add summary
    tf.compat.v1.summary.scalar('learning_rate', learning_rate)
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate)
    d_loss, r_loss, model_loss = compute_loss(f_score, f_geometry, recognition_logits, input_score_maps, input_geo_maps,
                                              input_training_masks, input_transcription, input_box_widths)
    tf.compat.v1.summary.scalar('total_loss', model_loss)
    total_loss = tf.add_n([model_loss] + tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
    batch_norm_updates_op = tf.group(*tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS))
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.compat.v1.summary.merge_all()
    variable_averages = tf.compat.v1.train.ExponentialMovingAverage(
        config.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.compat.v1.trainable_variables())
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=1)
    summary_writer = tf.compat.v1.summary.FileWriter(config.checkpoint_path, tf.compat.v1.get_default_graph())

    init = tf.compat.v1.global_variables_initializer()

    if config.pre_trained_model_path is not None:
        if os.path.isdir(config.pre_trained_model_path):
            print("Restore pretrained model from other datasets")
            ckpt = tf.compat.v1.train.latest_checkpoint(config.pre_trained_model_path)
            variable_restore_op = slim.assign_from_checkpoint_fn(ckpt, slim.get_trainable_variables(),
                                                                 ignore_missing_vars=True)
        else:  # is *.ckpt
            print("Restore pretrained model from imagenet")
            variable_restore_op = slim.assign_from_checkpoint_fn(config.pre_trained_model_path,
                                                                 slim.get_trainable_variables(),
                                                                 ignore_missing_vars=True)

    with tf.compat.v1.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if config.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(config.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if config.pre_trained_model_path is not None:
                variable_restore_op(sess)

        data_generator = dataset.get_batch(num_workers=10,
                                           input_size=config.input_size,
                                           batch_size=config.batch_size_per_gpu)
        tf.compat.v1.train.write_graph(tf.get_default_graph(), 'weights/', 'tf_graph.pb', as_text=False)
        print('--- Training ---')
        print("Parameters : {}".format(get_parameter()))
        for step in range(config.max_steps):
            data = next(data_generator)
            inp_dict = {input_images: data[0],
                        input_score_maps: data[2],
                        input_geo_maps: data[3],
                        input_training_masks: data[4],
                        input_transform_matrix: data[5],
                        input_box_widths: data[7],
                        input_transcription: data[8]}

            for i in range(config.batch_size_per_gpu):
                inp_dict[input_box_masks[i]] = data[6][i]

            dl, rl, tl, _ = sess.run([d_loss, r_loss, total_loss, train_op], feed_dict=inp_dict)
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break

            if step % 10 == 0:
                print('Epoch {:06d}, d_loss {:.4f}, r_loss {:.4f}, t_loss {:.4f},'.format(step, dl, rl, tl))

            if step % config.save_checkpoint_steps == 0:
                saver.save(sess, config.checkpoint_path + 'model.ckpt', global_step=global_step)

            if step % config.save_summary_steps == 0:
                dl, rl, tl, _, summary_str = sess.run([d_loss,
                                                       r_loss,
                                                       total_loss,
                                                       train_op,
                                                       summary_op],
                                                      feed_dict=inp_dict)

                summary_writer.add_summary(summary_str, global_step=step)


if __name__ == '__main__':
    main()
