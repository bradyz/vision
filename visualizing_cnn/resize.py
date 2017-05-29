'''
Resize a JPG.
'''
import argparse

from PIL import Image


def resize(filename, width, height):
    image = Image.open(filename)
    image = image.resize((width, height), Image.ANTIALIAS)

    filename_without_extension = ''.join(filename.split('.')[:-1])

    out_path = '%s_%d_%d.jpg' % (filename_without_extension, width, height)

    image.save(out_path, 'jpeg')

    print('Saved to %s' % out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=str)
    parser.add_argument('width', type=int)
    parser.add_argument('height', type=int)

    args = parser.parse_args()
    filename = args.filename
    width = args.width
    height = args.height

    resize(filename, width, height)
