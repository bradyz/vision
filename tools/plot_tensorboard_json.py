import json

import numpy as np

import seaborn
import matplotlib.pyplot as plt


def load_json(filename):
    with open(filename, 'r') as fd:
        result = json.load(fd)

    result = np.float32(result)

    # Ignore wall clock time.
    result = result[:,1:]
    result[:,0] = np.arange(result.shape[0])

    return result


def smooth_curve(x_y, k):
    result = np.copy(x_y)
    result[:,0] = x_y[:,0]
    result[:,1] = np.convolve(x_y[:,1], np.ones(k) / k, mode='same')
    result[:k,1] = x_y[:k,1]
    result[-k:,1] = x_y[-k:,1]

    return result


def plot(x_y, label=None):
    plt.plot(x_y[:,0], x_y[:,1], '-', label=label)


accs = dict()
accs['kmeans_adam'] = 'run_bn_kmeans_256_2_angle_e1_full_adam-tag-Vanilla-metrics-accuracy.json'
accs['kmeans_sgd'] = 'run_bn_kmeans_256_2_angle_e1_full_sgd-tag-Vanilla-metrics-accuracy.json'
accs['he_sgd'] = 'run_bn_he_sgd-tag-Vanilla-metrics-accuracy.json'
accs['he_adam'] = 'run_bn_he_adam-tag-Vanilla-metrics-accuracy.json'
accs['xavier_sgd'] = 'run_bn_xavier_sgd-tag-Vanilla-metrics-accuracy.json'
accs['xavier_adam'] = 'run_bn_xavier_adam-tag-Vanilla-metrics-accuracy.json'
accs['std_sgd'] = 'run_bn_std_sgd-tag-Vanilla-metrics-accuracy.json'
accs['std_adam'] = 'run_bn_std_adam-tag-Vanilla-metrics-accuracy.json'

losses = dict()
losses['kmeans_adam'] = 'run_bn_kmeans_256_2_angle_e1_full_adam-tag-Vanilla-loss-xent.json'
losses['kmeans_sgd'] = 'run_bn_kmeans_256_2_angle_e1_full_sgd-tag-Vanilla-loss-xent.json'
losses['he_sgd'] = 'run_bn_he_sgd-tag-Vanilla-loss-xent.json'
losses['he_adam'] = 'run_bn_he_adam-tag-Vanilla-loss-xent.json'
losses['xavier_sgd'] = 'run_bn_xavier_sgd-tag-Vanilla-loss-xent.json'
losses['xavier_adam'] = 'run_bn_xavier_adam-tag-Vanilla-loss-xent.json'
losses['std_sgd'] = 'run_bn_std_sgd-tag-Vanilla-loss-xent.json'
losses['std_adam'] = 'run_bn_std_adam-tag-Vanilla-loss-xent.json'

accs_test = dict()
accs_test['kmeans_adam'] = 'run_bn_kmeans_256_2_angle_e2_adam-tag-val-Vanilla-metrics-accuracy.json'
accs_test['kmeans_sgd'] = 'run_bn_kmeans_64_2_angle_e2_sgd-tag-val-Vanilla-metrics-accuracy.json'
accs_test['he_sgd'] = 'run_bn_he_sgd-tag-val-Vanilla-metrics-accuracy.json'
accs_test['he_adam'] = 'run_bn_he_adam-tag-val-Vanilla-metrics-accuracy.json'
accs_test['xavier_sgd'] = 'run_bn_xavier_sgd-tag-val-Vanilla-metrics-accuracy.json'
accs_test['xavier_adam'] = 'run_bn_xavier_adam-tag-val-Vanilla-metrics-accuracy.json'
accs_test['std_sgd'] = 'run_bn_std_sgd-tag-val-Vanilla-metrics-accuracy.json'
accs_test['std_adam'] = 'run_bn_std_adam-tag-val-Vanilla-metrics-accuracy.json'

loss_test = dict()
loss_test['kmeans_adam'] = 'run_bn_kmeans_256_2_angle_e2_adam-tag-val-Vanilla-loss-xent.json'
loss_test['kmeans_sgd'] = 'run_bn_kmeans_64_2_angle_e2_sgd-tag-val-Vanilla-loss-xent.json'
loss_test['he_sgd'] = 'run_bn_he_sgd-tag-val-Vanilla-loss-xent.json'
loss_test['he_adam'] = 'run_bn_he_adam-tag-val-Vanilla-loss-xent.json'
loss_test['xavier_sgd'] = 'run_bn_xavier_sgd-tag-val-Vanilla-loss-xent.json'
loss_test['xavier_adam'] = 'run_bn_xavier_adam-tag-val-Vanilla-loss-xent.json'
loss_test['std_sgd'] = 'run_bn_std_sgd-tag-val-Vanilla-loss-xent.json'
loss_test['std_adam'] = 'run_bn_std_adam-tag-val-Vanilla-loss-xent.json'

maxs = dict()
mins = dict()

for key, path in accs_test.items():
    print(key)

    # accs_test[key] = smooth_curve(load_json(path), 3)
    accs_test[key] = load_json(path)

    if 'adam' in key:
        maxs[key] = np.max(accs_test[key][:,1])
        mins[key] = np.min(accs_test[key][:,1])

import pdb; pdb.set_trace()


for key, curve in losses.items():
    plot(curve, key)

plt.grid(linestyle='--')
plt.legend()
plt.title('Training Loss')
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epochs')
plt.show()

