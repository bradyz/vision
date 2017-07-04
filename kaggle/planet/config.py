train_path = 'train_v2.csv'
image_dir = 'train-jpg'
model_path = 'model_dense.h5'

num_classes = 17
batch_size = 128

image_shape = [96, 96, 3]

FEATURES = ['activation_1', 'activation_10', 'activation_22',
            'activation_40', 'activation_49']

image_mean = [77.09, 87.68, 80.24]
image_std = [35.84, 37.44, 43.15]
