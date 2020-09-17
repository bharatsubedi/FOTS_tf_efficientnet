import os
import time

import cv2
import numpy as np
import tensorflow as tf

from utils import config
from utils import dataset
from utils import rotate
from Network import detection, recognition
from utils import lanms
#import lanms
os.environ["CUDA_VISIBLE_DEVICES"]="0"
detect_part = detection.Detection(is_training=False)
roi_rotate_part = rotate.RoIRotate()
recognize_part = recognition.Recognition(is_training=False)
font = cv2.FONT_HERSHEY_SIMPLEX

gt = 'Inferencing'
compare_file = open("{}.txt".format(gt), "w")
def resize_image(im, max_side_len=800):
    h, w = im.shape
    resize_w = w
    resize_h = h
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)
    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_threshold=0.2):
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    xy_text = np.argwhere(score_map > score_map_thresh)
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    start = time.time()
    text_box_restored = dataset.restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    start = time.time()
    #boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_threshold)
    boxes = lanms.nms_locality(boxes.astype('float32'), nms_threshold)
    timer['nms'] = time.time() - start
    if boxes.shape[0] == 0:
        return None, timer
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]
    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]
####### matrix generation ##################
def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    #print (matrix)
    return matrix, matrix[-1][-1]    
###### matrix generation and counting wrong string end##################
###### comparing string and finding wrong string and character ####

def stringfinding(m,seq1,seq2):
    str_result = [char for char in seq2]
    str_gt = [char1 for char1 in seq1]
    print(str_result)
    print(str_gt)
    index1 = 0
    index2 = 0
    match_char=0
    for a in reversed(range(len(seq1) + 1 + index1)):
        for b in reversed(range(len(seq2) + 1 + index2)):
            if a > 0 and b > 0:
                min_value = min(m[a - 1][b], m[a - 1][b - 1], m[a][b - 1])
                ab = m[a - 1][b]
                ac = m[a - 1][b - 1]
                ad = m[a][b - 1]
                if min_value == m[a - 1][b - 1]:
                    a_value = seq1[a - 1]
                    b_value = seq2[b - 1]
                    if seq1[a - 1] == seq2[b - 1]:
                        index1 += -1
                        index2 += -1
                        match_char+=1
                        compare_file.write("matching char::{}\n".format(seq1[a - 1]))
                        print("matching chararacter::", seq1[a - 1])
                        break
                    else:
                        index1 += -1
                        index2 += -1
                        compare_file.write("recognition character::{} to be replace by ground truth::{}\n".format(seq2[b - 1],seq1[a - 1]))
                        print("recognition character::{} to be replace by ground truth::{}".format(seq2[b - 1],seq1[a - 1]))
                        break
                else:
                    if min_value == m[a][b - 1]:
                        index1 += 0
                        index2 += -1
                        compare_file.write("delete recognition character to match ground truth::{}\n".format(seq2[b - 1]))
                        print("delete recognition character to match ground truth::", seq2[b - 1])
                    if min_value == m[a - 1][b]:
                        index1 += -1
                        index2 += 0
                        compare_file.write("add new character to match ground truth::{}\n".format(seq1[a - 1]))
                        print("add new character to match ground truth::", seq1[a - 1])
                        break
    return match_char
######## finding string and wrong character end ############

def main():
    with tf.get_default_graph().as_default():
        ###### testing file ###################
        
        count_string = 0
        wrong_string =0
        total_match_char = 0
        total_wrong_char = 0
        
        ########## testing file opening end ##############
        
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_images')
        input_feature_map = tf.placeholder(tf.float32, shape=[None, None, None, 32], name='input_feature_map')
        input_transform_matrix = tf.placeholder(tf.float32, shape=[None, 6], name='input_transform_matrix')
        input_box_mask = [tf.placeholder(tf.int32, shape=[None], name='input_box_masks_0')]
        input_box_widths = tf.placeholder(tf.int32, shape=[None], name='input_box_widths')
        shared_feature, f_score, f_geometry = detect_part.build_graph(input_images)
        pad_rois = roi_rotate_part.roi_rotate_tensor_pad(input_feature_map,
                                                         input_transform_matrix,
                                                         input_box_mask,
                                                         input_box_widths)
        recognition_logits, dense_decode = recognize_part.model(pad_rois, input_box_widths)
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        stime=time.time()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(config.checkpoint_path)
            model_path = os.path.join(config.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
            #tf.train.write_graph(sess.graph_def, 'weights/', 'tf_graph.pb')
            image_paths = dataset.get_images(path=config.test_data_path)
            count_char=0
            for image_path in image_paths:
                image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                print(image_path)
                gt_name = image_path[13:20]
                #name = image_path[13:-4]
                #gt_name='gt_'+name+'.txt'
                print(image_path)
                #print(gt_name)
                #file_gt = open('dataset/gt/{}'.format(gt_name), 'r')
                #gt_string = file_gt.read()
                gt_string=gt_name
                #print("gt_string", gt_string)
                start_time = time.time()
                #image = cv2.resize(image,(800,600))
                im_resized, (ratio_h, ratio_w) = resize_image(image)
                im_h, im_w = im_resized.shape
                im_resized = im_resized.reshape(im_h, im_w, 1)
                timer = {'detect': 0, 'restore': 0, 'nms': 0, 'recognition': 0}
                start = time.time()
                bounding = []

                shared_feature_map, score, geometry = sess.run([shared_feature, f_score, f_geometry],
                                                               feed_dict={input_images: [im_resized]})
                boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                timer['detect'] = time.time() - start
                start = time.time()
                # save to file
                if boxes is not None and boxes.shape[0] != 0:
                    res_file_path = os.path.join(config.output_dir,
                                                 'res_' + '{}.txt'.format(os.path.basename(image_path).split('.')[0]))

                    input_roi_boxes = boxes[:, :8].reshape(-1, 8)
                    r_decode_list = []
                    for batch_index in range(input_roi_boxes.shape[0] // 32 + 1):  # test roi batch size is 32
                        start_slice_index = batch_index * 32
                        end_slice_index = (batch_index + 1) * 32 if input_roi_boxes.shape[0] >= (
                                batch_index + 1) * 32 else input_roi_boxes.shape[0]
                        tmp_roi_boxes = input_roi_boxes[start_slice_index:end_slice_index]

                        boxes_masks = [0] * tmp_roi_boxes.shape[0]
                        transform_matrices, box_widths = dataset.get_project_matrix_and_width(tmp_roi_boxes)
                        r_decode = sess.run(dense_decode, feed_dict={input_feature_map: shared_feature_map,
                                                                     input_transform_matrix: transform_matrices,
                                                                     input_box_mask[0]: boxes_masks,
                                                                     input_box_widths: box_widths})
                        r_decode_list.extend([r for r in r_decode])

                    timer['recognition'] = time.time() - start
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                    if len(r_decode_list) != boxes.shape[0]:
                        print("detection and recognition result are not equal!")
                        exit(-1)

                    with open(res_file_path, 'w') as f:
                        for i, box in enumerate(boxes):
                            bound = []
                            box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                                continue
                            recognition_result = dataset.ground_truth_to_word(r_decode_list[i])
                            cor1 = int(box[0, 0])
                            cor2 = int(box[0, 1])
                            cor3 = int(box[2, 0])
                            cor4 = int(box[2, 1])
                            bound.append(cor1)
                            bound.append(cor2)
                            bound.append(cor3)
                            bound.append(cor4)
                            bound.append(recognition_result)
                            bounding.append(bound)
                        vals_x1 = sorted((el for el in bounding if el[0]>-1), key=lambda L: L[0])
                        vals_Y1 = sorted((el for el in bounding if el[0]>-1), key=lambda L: L[1])
                        vals_x2 = sorted((el for el in bounding if el[0]>-1), key=lambda L: L[2]) 
                        vals_Y2 = sorted((el for el in bounding if el[0]>-1), key=lambda L: L[3]) 
                        ls = [l[4] for l in vals_x1]
                        string = ''.join(ls)
                        if len(gt_string)<len(string):
                            string=string[:7]
                            print("extra::", string)
                        ### call matrix generation and count wrong string and character ###########
                        if string==gt_string:
                            count_string+=1
                        else:
                            wrong_string+=1
                            
                        
                        m,abc=levenshtein(gt_string, string)
                        compare_file.write("----------{}-------------\n".format(image_path[13:]))
                        compare_file.write("ground truth::{}----recognition result::{}\n".format(gt_string,string))
                        total_wrong_char+=abc                        
                        if abc>0:
                            match_char=stringfinding(m,gt_string,string)
                            total_match_char+=match_char
                            compare_file.write("match character number::{}\n".format(match_char))
                            compare_file.write("wrong character number::{}\n".format(abc))
                            print(abc)
                            print(image_path[13:])
                        else:
                            match_char=stringfinding(m,gt_string,string)
                            total_match_char+=match_char
                            compare_file.write("match character number::{}\n".format(match_char))
                            compare_file.write("wrong character number::{}\n".format(abc))
                         ###end call matrix generation and count wrong string and character ###########
                        x1 = 0 
                        y1 =0
                        for i, j in enumerate(vals_x1[0]):
                            if i == 0:
                                x1=j
                        for i, j in enumerate(vals_Y1[0]):
                            if i == 1:
                                y1=j
                        y2 = 0
                        x2 =0
                        for i, j in enumerate(vals_x2[len(vals_Y2)-1]):
                            if i ==2:
                                x2 =j
                        for i, j in enumerate(vals_Y2[len(vals_Y2)-1]):
                            if i ==3:
                                y2 =j
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2) 
                        
                        font_scale = 1.5
                        rectangle_bgr = (255, 255, 255)
                        (text_width, text_height) = cv2.getTextSize(string, font, fontScale=font_scale, thickness=1)[0]
                        # set the text start position
                        text_offset_x = x1
                        text_offset_y = y1
                        # make the coords of the box with a small padding of two pixels
                        #box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
                       # cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
                        #im_txt = cv2.putText(image, string, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
                        #f.write('{},{},{},{},{}\r\n'.format(x1,y1,x2,y2,string))
                else:
                    timer['net'] = time.time() - start
                    #res_file = os.path.join(config.output_dir,
                    #                        'res_' + '{}.txt'.format(os.path.basename(image_path).split('.')[0]))
                    #f = open(res_file, "w")
                    #f.close()
                duration = time.time() - start_time
                print('[timing] {}'.format(duration))
                #img_path = os.path.join(config.output_dir, os.path.basename(image_path))
                #cv2.imwrite(img_path, im_txt)
    etime=time.time()
    dura=etime-stime
    total_character=total_wrong_char+total_match_char
    char_accuracy=(total_match_char/total_character)*100
    total_string = wrong_string+count_string
    string_accuracy=(count_string/total_string)*100
    compare_file.write("------------total compare resutl---------------------------\n")
    compare_file.write("total wrong character number::{}\n".format(total_wrong_char))
    compare_file.write("total correct character number::{}\n".format(total_match_char))
    compare_file.write("total wrong string number::{}\n".format(wrong_string))
    compare_file.write("total correct string number::{}\n".format(count_string))
    compare_file.write("total character number::{}\n".format(total_character))
    compare_file.write("character accuracy::{}\n".format(char_accuracy))
    compare_file.write("total string number::{}\n".format(total_string))
    compare_file.write("string accuracy::{}\n".format(string_accuracy))
    compare_file.write("total time::{}\n".format(dura))
if __name__ == '__main__':
    main()
