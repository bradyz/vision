import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

from dataset import CIFAR10
from logger import Logger
from model import BasicNetwork
from utils import maybe_load, save
from parameters import Parameters


def modify_learning_rate(epoch, optimizer, params):
    alpha = epoch / params.max_epoch

    lr = (1.0 - alpha) * params.lr + alpha * params.end_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_initial_mean_std(weights):
    means = list()
    stds = list()

    for weight in weights:
        means.append(torch.mean(weight.data))
        stds.append(torch.std(weight.data))

    return list(zip(means, stds))


def kl_divergence(mu_1, sigma_1, mu_2, sigma_2, eps=1e-7):
    lhs = torch.log(sigma_1 / sigma_2 + eps)
    rhs = (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / (2 * (sigma_2 ** 2) + eps) + 0.5

    return lhs + rhs


class CrossEntropyKLLoss(object):
    def __init__(self, weights, decay):
        self.weights = weights
        self.decay = decay

        self.loss = nn.CrossEntropyLoss()

        self.initial_mean_std = get_initial_mean_std(weights)

    def __call__(self, logits, targets):
        xent_loss = self.loss(logits, targets)

        reg_loss = 0.0

        for weight, (mean, std) in zip(self.weights, self.initial_mean_std):
            reg_loss += kl_divergence(mean, std, torch.mean(weight), torch.std(weight))

        total_loss = xent_loss + self.decay * reg_loss

        return total_loss, xent_loss, reg_loss


def train_or_test(net, opt, crit, log, data, params, is_train, is_first):
    if is_train:
        net.train()
    else:
        net.eval()

    # Metrics.
    losses = list()
    reg_losses = list()

    correct = 0
    total_samples = 0

    for inputs, targets in tqdm.tqdm(data, total=len(data), desc='Batch'):
        if params.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs), Variable(targets)

        if is_train:
            opt.zero_grad()

        logits = net(inputs)
        total_loss, xent_loss, reg_loss = crit(logits, targets)

        if is_train and not is_first:
            total_loss.backward()

            opt.step()

        losses.append(xent_loss.cpu().data[0])
        reg_losses.append(reg_loss.cpu().data[0])

        _, predicted = torch.max(logits.data, 1)
        correct += predicted.eq(targets.data).cpu().sum()

        total_samples += targets.size(0)

    loss = np.mean(losses)
    reg_loss = np.mean(reg_losses)
    accuracy = correct / total_samples

    log.update(reg_loss, loss, accuracy, is_train)


def main(params):
    scope = list()

    net = BasicNetwork(CIFAR10.in_channels, CIFAR10.num_classes, scope)
    log = Logger()
    opt = optim.SGD(
            net.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    state = {'net': net,
            'log': log,
            'opt': opt,
            'params': params}

    maybe_load(state, params.checkpoint_path)

    print(params)

    log.visible = params.use_vizdom

    crit = CrossEntropyKLLoss(scope, params.kl_decay)

    if params.use_cuda:
        net.cuda()

    data_train = CIFAR10.get_data(params.data_dir, True, params.batch_size)
    data_test = CIFAR10.get_data(params.data_dir, False, params.batch_size)

    for epoch in tqdm.trange(log.epoch, params.max_epoch + 1, desc='Epoch'):
        modify_learning_rate(epoch, opt, params)

        is_first = epoch == 0

        train_or_test(
                net, opt, crit, log, data_train, params, True, is_first)

        train_or_test(
                net, opt, crit, log, data_test, params, False, is_first)

        log.set_epoch(epoch+1)

        if log.should_save():
            save(state, params.checkpoint_path)


if __name__ == '__main__':
    params = Parameters()
    params.parse()

    main(params)
