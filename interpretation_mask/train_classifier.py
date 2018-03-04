import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from torch.autograd import Variable

from dataset import CIFAR10
from logger import Logger
from model import ResNet18, MaskGenerator
from utils import maybe_load, save
from parameters import Parameters


class AttentionMaskLoss(object):
    def __init__(self, alpha, beta, max_norm):
        self.alpha = alpha
        self.beta = beta
        self.max_norm = Variable(torch.FloatTensor([max_norm])).cuda()

        self.xent_loss = nn.CrossEntropyLoss()

    def __call__(self, mask, foreground_logits, background_logits, targets):
        xent_loss = self.xent_loss(foreground_logits, targets)

        background_prob = nn.functional.softmax(background_logits, 1)
        background_entropy = (background_prob * torch.log(background_prob)).sum(1).mean()

        mask_flattened = mask.view(mask.size(0), -1)
        mask_norm = mask_flattened.norm(2, 1)

        mask_norm_sq_loss = (mask_norm - self.max_norm) ** 2
        mask_norm_sq_loss = mask_norm_sq_loss.mean()

        # mask_norm_sq_loss = (mask_norm > self.max_norm).float() * mask_norm
        # mask_norm_sq_loss = (mask_norm_sq_loss ** 2).mean()

        total_loss = xent_loss
        total_loss -= self.alpha * background_entropy
        total_loss += self.beta * mask_norm_sq_loss

        return total_loss


def train_or_test(classifier, optimizer_classifier, masker, optimizer_masker,
        criterion, logger, data, params, is_train):
    if is_train:
        classifier.train()
        masker.train()
    else:
        classifier.eval()
        masker.eval()

    # Metrics.
    losses_mask = list()
    losses = list()

    correct = 0
    total_samples = 0

    for inputs, targets in tqdm.tqdm(data, total=len(data)):
        if params.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs), Variable(targets)

        if is_train:
            optimizer_classifier.zero_grad()
            optimizer_masker.zero_grad()

        mask = torch.round(masker(inputs))
        mask_rgb = torch.cat([mask, mask, mask], 1)

        foreground = mask_rgb * inputs
        background = (1.0 - mask_rgb) * inputs

        foreground_logits = classifier(foreground)
        background_logits = classifier(background)

        mask_loss = criterion(mask, foreground_logits, background_logits, targets)

        losses_mask.append(mask_loss.cpu().data[0])

        if is_train:
            mask_loss.backward()
            optimizer_masker.step()

            optimizer_masker.zero_grad()
            optimizer_classifier.zero_grad()

        logits = classifier(inputs)
        loss = criterion.xent_loss(logits, targets)

        if is_train:
            loss.backward()

            optimizer_classifier.step()

        logger.draw(
                inputs.cpu().data.numpy(),
                mask_rgb.cpu().data.numpy(),
                is_train)

        losses.append(loss.cpu().data[0])

        _, predicted = torch.max(logits.data, 1)
        correct += predicted.eq(targets.data).cpu().sum()

        total_samples += targets.size(0)

    loss = np.mean(losses)
    losses_mask = np.mean(losses_mask)

    accuracy = correct / total_samples

    logger.update(loss, accuracy, losses_mask, is_train)


def main(params):
    classifier = ResNet18(CIFAR10.in_channels)
    masker = MaskGenerator(CIFAR10.in_channels)
    logger = Logger()
    optimizer_classifier = optim.Adam(
            classifier.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    optimizer_masker = optim.Adam(
            masker.parameters(), lr=1e-5, weight_decay=params.weight_decay)

    state = {'classifier': classifier,
            'masker': masker,
            'logger': logger,
            'optimizer_classifier': optimizer_classifier,
            'optimizer_masker': optimizer_masker,
            'params': params}

    maybe_load(state, params.checkpoint_path)

    print(params)

    logger.visible = params.use_vizdom

    criterion = AttentionMaskLoss(params.alpha, params.beta, params.max_norm)

    if params.use_cuda:
        classifier.cuda()
        masker.cuda()

    data_train = CIFAR10.get_data(params.data_dir, True, params.batch_size)
    data_test = CIFAR10.get_data(params.data_dir, False, params.batch_size)

    for epoch in range(logger.epoch, params.max_epoch):
        print('Epoch: %s' % epoch)

        train_or_test(
                classifier, optimizer_classifier, masker, optimizer_masker,
                criterion, logger, data_train, params, True)

        train_or_test(
                classifier, optimizer_classifier, masker, optimizer_masker,
                criterion, logger, data_test, params, False)

        logger.set_epoch(epoch+1)

        if logger.should_save():
            save(state, params.checkpoint_path)


if __name__ == '__main__':
    params = Parameters()
    params.parse()

    main(params)
