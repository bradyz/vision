train_path = 'train_v2.csv'
image_dir = '/Users/bradyzhou/code/vision/kaggle/planet/train-jpg'

num_classes = 18
batch_size = 32

image_shape = [224, 224, 3]

FEATURES = ['block1_pool', 'block2_pool', 'block3_pool',  'block4_pool']
