import tensorflow as tf


def transformer(input_f_map, theta, out_dims=None):
    # grab input dimensions
    B = tf.shape(input_f_map)[0]
    H = tf.shape(input_f_map)[1]
    W = tf.shape(input_f_map)[2]

    # reshape theta to (B, 2, 3)
    theta = tf.reshape(theta, [B, 2, 3])

    # generate grids of same size or up-sample/down-sample if specified
    if out_dims:
        out_H = out_dims[0]
        out_W = out_dims[1]
        batch_grids = affine_grid_generator(out_H, out_W, theta)
    else:
        batch_grids = affine_grid_generator(H, W, theta)

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    # sample input with grid to get output
    out_f_map = bi_linear_sampler(input_f_map, x_s, y_s)

    return out_f_map


def get_pixel_value(img, x, y):
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def affine_grid_generator(height, width, theta):
    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid
    x = tf.linspace(0.0, 1.0, width)
    y = tf.linspace(0.0, 1.0, height)
    x = x * tf.cast(width, tf.float32)
    y = y * tf.cast(height, tf.float32)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids


def bi_linear_sampler(img, x, y):
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    # x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    # y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out


class RoIRotate(object):
    def __init__(self, height=8):
        self.height = height

    @staticmethod
    def roi_rotate_tensor(feature_map, transform_matrices, box_masks, box_widths):
        with tf.variable_scope("RoI_Rotate"):
            max_width = box_widths[tf.argmax(box_widths, 0, output_type=tf.int32)]
            box_widths = tf.cast(box_widths, tf.float32)
            tile_feature_maps = []
            map_shape = tf.shape(feature_map)
            map_shape = tf.cast(map_shape, tf.float32)

            for i, mask in enumerate(box_masks):  # box_masks is a list of num of rois in each feature map
                _feature_map = feature_map[i]
                _feature_map = tf.expand_dims(_feature_map, axis=0)
                box_nums = tf.shape(mask)[0]
                _feature_map = tf.tile(_feature_map, [box_nums, 1, 1, 1])
                tile_feature_maps.append(_feature_map)

            tile_feature_maps = tf.concat(tile_feature_maps, axis=0)  # N' * H * W * C where N' = N * B
            norm_box_widths = box_widths / map_shape[2]
            ones = tf.ones_like(norm_box_widths)
            norm_box_heights = ones * (8.0 / map_shape[1])
            zeros = tf.zeros_like(norm_box_widths)
            crop_boxes = tf.transpose(tf.stack([zeros, zeros, norm_box_heights, norm_box_widths]))
            crop_size = tf.transpose(tf.stack([8, max_width]))

            trans_feature_map = transformer(tile_feature_maps, transform_matrices)

            box_indexes = tf.range(tf.shape(trans_feature_map)[0])
            rois = tf.image.crop_and_resize(trans_feature_map, crop_boxes, box_indexes, crop_size)

            pad_rois = tf.image.pad_to_bounding_box(rois, 0, 0, 8, max_width)
            return pad_rois

    @staticmethod
    def roi_rotate_tensor_pad(feature_map, transform_matrices, box_masks, box_widths):
        with tf.variable_scope("RoI_Rotate"):
            max_width = box_widths[tf.argmax(box_widths, 0, output_type=tf.int32)]
            tile_feature_maps = []

            for i, mask in enumerate(box_masks):
                _feature_map = feature_map[i]
                _feature_map = tf.expand_dims(_feature_map, axis=0)
                box_nums = tf.shape(mask)[0]
                _feature_map = tf.tile(_feature_map, [box_nums, 1, 1, 1])
                tile_feature_maps.append(_feature_map)

            tile_feature_maps = tf.concat(tile_feature_maps, axis=0)  # N' * H * W * C where N' = N * B
            trans_feature_map = transformer(tile_feature_maps, transform_matrices)

            box_nums = tf.shape(box_widths)[0]
            pad_rois = tf.TensorArray(tf.float32, box_nums)
            i = 0

            def cond(_pad_rois, _i):
                return _i < box_nums

            def body(_pad_rois, _i):
                _affine_feature_map = trans_feature_map[_i]
                width_box = box_widths[_i]
                roi = tf.image.crop_to_bounding_box(_affine_feature_map, 0, 0, 8, width_box)
                pad_roi = tf.image.pad_to_bounding_box(roi, 0, 0, 8, max_width)
                _pad_rois = _pad_rois.write(_i, pad_roi)
                _i += 1

                return _pad_rois, _i
            pad_rois, _ = tf.while_loop(cond, body, loop_vars=[pad_rois, i])
            pad_rois = pad_rois.stack()
            return pad_rois

    @staticmethod
    def roi_rotate_tensor_while(feature_map, transform_matrices, box_masks, box_widths):
        assert transform_matrices.shape[-1] != 8
        with tf.variable_scope("RoI_Rotate"):
            box_masks = tf.concat(box_masks, axis=0)
            box_nums = tf.shape(box_widths)[0]
            pad_rois = tf.TensorArray(tf.float32, box_nums)
            max_width = box_widths[tf.arg_max(box_widths, 0, tf.int32)]
            i = 0

            def cond(_pad_rois, _i):
                return _i < box_nums

            def body(_pad_rois, _i):
                index = box_masks[_i]
                matrix = transform_matrices[_i]
                _feature_map = feature_map[index]
                map_shape = tf.shape(_feature_map)
                map_shape = tf.to_float(map_shape)
                width_box = box_widths[_i]
                width_box = tf.cast(width_box, tf.float32)

                after_transform = tf.contrib.image.transform(_feature_map, matrix, "BILINEAR")
                after_transform = tf.expand_dims(after_transform, 0)
                roi = tf.image.crop_and_resize(after_transform,
                                               [[0, 0, 8 / map_shape[0], width_box / map_shape[1]]],
                                               [0],
                                               [8, tf.cast(width_box, tf.int32)])
                pad_roi = tf.image.pad_to_bounding_box(roi, 0, 0, 8, max_width)
                _pad_rois = _pad_rois.write(_i, pad_roi)
                _i += 1
                return _pad_rois, _i

            pad_rois, _ = tf.while_loop(cond, body, loop_vars=[pad_rois, i])
            pad_rois = pad_rois.stack()
            pad_rois = tf.squeeze(pad_rois, axis=1)
            return pad_rois
