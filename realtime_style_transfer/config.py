input_shape = [256, 256, 3]

checkpoint_steps = 10
save_steps = 1000
num_steps = 100000

batch_size = 6

vgg_weights = '/home/brady/data/imagenet-vgg-verydeep-19.mat'
model_name = 'style_transfer'
log_dir = 'less'

style_dir = '/home/brady/data/style_images/'
content_dir = '/home/brady/data/pascal_voc/VOCdevkit/VOC2009/JPEGImages/'

content_layers = ['relu2_1']
style_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
gram_weights = [1e0, 1e0, 1e0, 1e0]
