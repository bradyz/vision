import argparse

import torch


class Parameters(object):
    TO_IGNORE = ['use_cuda']

    def __init__(self):
        self.use_cuda = torch.cuda.is_available()

        self.checkpoint_path = None
        self.data_dir = None

        self.batch_size = 64
        self.lr = 2e-4
        self.max_epoch = 200

        self.weight_decay = 5e-5

        self.use_vizdom = True

        self.alpha = 0.01
        self.beta = 600.0

        self.max_norm = 16.0

    def parse(self):
        parser = argparse.ArgumentParser()

        for key, val in self.__dict__.items():
            if key in self.TO_IGNORE:
                continue

            parser.add_argument('--%s' % key, required=val is None)

        for key, val in parser.parse_args().__dict__.items():
            if val is None:
                continue
            elif self.__dict__[key] is None:
                self.__dict__[key] = val
            else:
                self.__dict__[key] = type(self.__dict__[key])(val)

    def __str__(self):
        result = list()

        for key, val in sorted(self.__dict__.items()):
            result.append('%s (%s): %s' % (key, val.__class__.__name__, val))

        return '\n'.join(result)

    def state_dict(self):
        return self.__dict__
