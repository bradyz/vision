import visdom
import numpy as np


from dataset import CIFAR10


class Logger(object):
    def __init__(self):
        self.epoch = 0

        self.viz = visdom.Visdom()

        self.reg_loss_train = ScalarPlot(self.viz, 'reg_loss_train')
        self.reg_loss_test = ScalarPlot(self.viz, 'reg_loss_test')

        self.loss_train = ScalarPlot(self.viz, 'loss_train')
        self.loss_test = ScalarPlot(self.viz, 'loss_test')

        self.accuracy_train = ScalarPlot(self.viz, 'accuracy_train')
        self.accuracy_test = ScalarPlot(self.viz, 'accuracy_test')

        self.use_vizdom = True

    def update(self, reg_loss, loss, accuracy, is_train):
        if is_train:
            self.reg_loss_train.draw(reg_loss, self.use_vizdom)
            self.loss_train.draw(loss, self.use_vizdom)
            self.accuracy_train.draw(accuracy, self.use_vizdom)
        else:
            self.reg_loss_test.draw(reg_loss, self.use_vizdom)
            self.loss_test.draw(loss, self.use_vizdom)
            self.accuracy_test.draw(accuracy, self.use_vizdom)

    def draw(self, is_train):
        if not self.use_vizdom:
            return

        self.reg_loss_train.draw()
        self.loss_train.draw()
        self.accuracy_train.draw()

        self.reg_loss_test.draw()
        self.loss_test.draw()
        self.accuracy_test.draw()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def should_save(self):
        return True
        # return min(self.test_loss) == self.test_loss[-1]

    def load_state_dict(self, state):
        self.epoch = state['epoch']

        self.reg_loss_train.values = state['reg_loss_train']
        self.loss_train.values = state['loss_train']
        self.accuracy_train.values = state['accuracy_train']

        self.reg_loss_test.values = state['reg_loss_test']
        self.loss_test.values = state['loss_test']
        self.accuracy_test.values = state['accuracy_test']

    def state_dict(self):
        state = dict()

        state['epoch'] = self.epoch

        state['reg_loss_train'] = self.reg_loss_train.values
        state['loss_train'] = self.loss_train.values
        state['accuracy_train'] = self.accuracy_train.values

        state['reg_loss_test'] = self.reg_loss_test.values
        state['loss_test'] = self.loss_test.values
        state['accuracy_test'] = self.accuracy_test.values

        return state


class ScalarPlot(object):
    def __init__(self, viz, title):
        self.viz = viz
        self.title = title

        self.values = list()

    def draw(self, y=None, should_render=False):
        if y:
            self.values.append(y)

        if not self.values:
            return
        elif not should_render:
            return

        self.viz.line(
                X=np.float32(list(range(len(self.values)))),
                Y=np.float32(self.values),
                name=self.title,
                win=self.title,
                opts=dict(title=self.title))

    def __getitem__(self, idx):
        return self.values[idx]
