train_path = 'train_v2.csv'
image_dir = 'train-jpg'
model_path = 'model.h5'

num_classes = 17
batch_size = 64

image_shape = [224, 224, 3]

FEATURES = ['block2_pool', 'block3_pool', 'block4_conv3', 'block5_conv3']
