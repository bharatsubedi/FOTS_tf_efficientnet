# coding:utf-8
CHAR_VECTOR = "12345ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CHAR_VECTOR) + 1

test_data_path = 'dataset/test/'
gpu_list = '0'
checkpoint_path = './weights/Pre_weight_E4/'
output_dir = 'results/'
training_data_path = './dataset/train_data/'
training_label_path = './dataset/gt/'
split_character = '/'
is_multiprocess = True,
max_image_large_side = 1600
max_text_size = 800
min_text_size = 10
min_crop_side_ratio = 0.1
geometry = 'RBOX'
text_scale = 512
input_size = 512
batch_size_per_gpu = 6
num_readers = 10
learning_rate = 0.0001
max_steps = 100001
moving_average_decay = 0.997
restore = False
save_checkpoint_steps = 1000
save_summary_steps = 100
pre_trained_model_path = None #'pretrained weight path'

