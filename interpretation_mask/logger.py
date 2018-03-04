import visdom
import numpy as np


from dataset import CIFAR10


class Logger(object):
    def __init__(self):
        self.epoch = 0

        self.viz = visdom.Visdom()

        self.mask_loss_train = ScalarPlot(self.viz, 'mask_loss_train')
        self.mask_loss_test = ScalarPlot(self.viz, 'mask_loss_test')

        self.loss_train = ScalarPlot(self.viz, 'loss_train')
        self.loss_test = ScalarPlot(self.viz, 'loss_test')

        self.accuracy_train = ScalarPlot(self.viz, 'accuracy_train')
        self.accuracy_test = ScalarPlot(self.viz, 'accuracy_test')

        self.use_vizdom = True

    def update(self, loss, accuracy, loss_mask, is_train):
        if is_train:
            self.loss_train.draw(loss)
            self.accuracy_train.draw(accuracy)
            self.mask_loss_train.draw(loss_mask)
        else:
            self.loss_test.draw(loss)
            self.accuracy_test.draw(accuracy)
            self.mask_loss_test.draw(loss_mask)

    def draw(self, inputs, mask_rgb, is_train):
        if not self.use_vizdom:
            return

        self.loss_train.draw()
        self.accuracy_train.draw()
        self.mask_loss_train.draw()

        self.loss_test.draw()
        self.accuracy_test.draw()
        self.mask_loss_test.draw()

        inputs = inputs.transpose(0, 2, 3, 1) * CIFAR10.std + CIFAR10.mean
        inputs = inputs.transpose(0, 3, 1, 2)

        foreground = mask_rgb * inputs

        self.viz.images(inputs, win='train_input' if is_train else 'test_input')
        self.viz.images(foreground, win='train_foreground' if is_train else 'test_foreground')
        self.viz.images(mask_rgb, win='train_mask' if is_train else 'test_mask')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def should_save(self):
        return True
        # return min(self.test_loss) == self.test_loss[-1]

    def load_state_dict(self, state):
        self.epoch = state['epoch']

        self.loss_train.values = state['loss_train']
        self.accuracy_train.values = state['accuracy_train']
        self.mask_loss_train.values = state['mask_loss_train']

        self.loss_test.values = state['loss_test']
        self.accuracy_test.values = state['accuracy_test']
        self.mask_loss_test.values = state['mask_loss_test']

    def state_dict(self):
        state = dict()

        state['epoch'] = self.epoch

        state['loss_train'] = self.loss_train.values
        state['accuracy_train'] = self.accuracy_train.values
        state['mask_loss_train'] = self.mask_loss_train.values

        state['loss_test'] = self.loss_test.values
        state['accuracy_test'] = self.accuracy_test.values
        state['mask_loss_test'] = self.mask_loss_test.values

        return state


class ScalarPlot(object):
    def __init__(self, viz, title):
        self.viz = viz
        self.title = title

        self.values = list()
        self.win = None

    def draw(self, y=None):
        if y:
            self.values.append(y)

        if not self.values:
            return

        if self.win is None:
            self.win = self.viz.line(
                    X=np.float32(list(range(len(self.values)))),
                    Y=np.float32(self.values),
                    name=self.title,
                    opts=dict(title=self.title))

        else:
            self.viz.line(
                    X=np.float32(list(range(len(self.values)))),
                    Y=np.float32(self.values),
                    name=self.title,
                    win=self.win,
                    opts=dict(title=self.title))

    def __getitem__(self, idx):
        return self.values[idx]
