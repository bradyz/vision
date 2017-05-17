import pickle

from matplotlib import pyplot as plt


batch_size = 16
image_shape = [32, 32, 3]
latent_n = 128
vae_alpha = 1000.0


def load_pickle(file_path):
    with open(file_path, 'rb') as fd:
        data = pickle.load(fd, encoding='bytes')
    return data


def show_image(image):
    plt.imshow(image)
    plt.show()
