from matplotlib import pyplot as plt
from matplotlib import patches

import numpy as np

class Labeler:
    def __init__(self, image, color='r'):
        self.image = image
        self.color = color

        self.sx = None
        self.sy = None

        self.dx = None
        self.dy = None

        fig = plt.figure()

        fig.canvas.mpl_connect('button_press_event', self.press)
        fig.canvas.mpl_connect('button_release_event', self.release)

        plt.imshow(image, cmap='gray', interpolation='nearest')
        plt.show()

    def get_data(self):
        return (self.sx, self.sy, self.dx, self.dy)

    def press(self, event):
        self.sx = int(event.xdata)
        self.sy = int(event.ydata)

    def release(self, event):
        self.dx = int(event.xdata - self.sx)
        self.dy = int(event.ydata - self.sy)

        plt.close()

    def show(self):
        _, axis = plt.subplots(1)
        axis.imshow(self.image, cmap='gray', interpolation='nearest')

        rect = patches.Rectangle((self.sx, self.sy), self.dx, self.dy,
                                 linewidth=1, edgecolor=self.color, fill=False)

        axis.add_patch(rect)

        plt.show()
