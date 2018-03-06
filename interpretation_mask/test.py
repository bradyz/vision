import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from torch.autograd import Variable

from dataset import CIFAR10
from logger import Logger
from model import ResNet18
from utils import maybe_load
from parameters import Parameters


class AttentionMaskLoss(object):
    def __init__(self, gamma, alpha, beta, reg, max_norm):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.reg = reg
        self.max_norm = Variable(torch.FloatTensor([max_norm])).cuda()

        self.xent_loss = nn.CrossEntropyLoss()

    def __call__(self, mask, foreground_logits, background_logits, targets):
        xent_loss = self.xent_loss(foreground_logits, targets)

        background_prob = nn.functional.softmax(background_logits, 1)
        background_entropy = (-background_prob * torch.log(background_prob)).sum(1).mean()

        reg_loss = self.reg * (
                torch.sum(torch.abs(mask[:, :, :, :-1] - mask[:, :, :, 1:])) +
                torch.sum(torch.abs(mask[:, :, :-1, :] - mask[:, :, 1:, :])))

        mask_flattened = mask.view(mask.size(0), -1)
        mask_norm = mask_flattened.norm(2, 1)

        # Absolute penalty.
        # mask_norm_sq_loss = mask_norm.mean()

        # Penalty from target mean.
        mask_norm_sq_loss = (mask_norm - self.max_norm) ** 2
        mask_norm_sq_loss = mask_norm_sq_loss.mean()

        # Squared penalty if greater than.
        # mask_norm_sq_loss = (mask_norm - self.max_norm) ** 2
        # mask_norm_sq_loss = (mask_norm > self.max_norm).float() * mask_norm_sq_loss
        # mask_norm_sq_loss = mask_norm_sq_loss.mean()

        total_loss = self.gamma * xent_loss
        total_loss -= self.alpha * background_entropy
        total_loss += self.beta * mask_norm_sq_loss
        total_loss += self.reg * reg_loss

        return total_loss, background_entropy.mean(), mask_norm.mean()


def test(inputs, targets, classifier, criterion, log, params, n_iterations=10000):
    classifier.eval()

    if params.use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()

    inputs, targets = Variable(inputs), Variable(targets)

    # mask = torch.zeros((inputs.size(0), 1, inputs.size(2), inputs.size(3)))
    mask = torch.rand((inputs.size(0), 1, inputs.size(2), inputs.size(3)))

    if params.use_cuda:
        mask = mask.cuda()

    mask = Variable(mask, requires_grad=True)

    optimizer = optim.Adam([mask], lr=1e-2)

    for _ in tqdm.trange(n_iterations, desc='Optimizing'):
        # mask_sig = torch.round(mask)
        mask_sig = torch.nn.functional.sigmoid(mask)
        mask_rgb = torch.cat([mask_sig, mask_sig, mask_sig], 1)

        foreground = mask_rgb * inputs
        background = (1.0 - mask_rgb) * inputs

        foreground_logits = classifier(foreground)
        background_logits = classifier(background)

        mask_loss, entropy, norm = criterion(
                mask_sig, foreground_logits, background_logits, targets)

        mask_loss.backward()

        optimizer.step()

        log.draw(
                inputs.cpu().data.numpy(),
                mask_rgb.cpu().data.numpy(),
                False)

        log.update(
                entropy.cpu().data[0],
                norm.cpu().data[0],
                mask_loss.cpu().data[0],
                False)


def main(params):
    classifier = ResNet18(CIFAR10.in_channels)
    log = Logger()

    state = {'classifier': classifier}

    maybe_load(state, params.checkpoint_path)

    criterion = AttentionMaskLoss(params.gamma, params.alpha, params.beta, params.reg, params.max_norm)

    if params.use_cuda:
        classifier.cuda()

    data = CIFAR10.get_data(params.data_dir, False, params.batch_size)

    for inputs, targets in tqdm.tqdm(data, total=len(data), desc='Dataset'):
        test(inputs, targets, classifier, criterion, log, params)


if __name__ == '__main__':
    params = Parameters()
    params.parse()

    main(params)
