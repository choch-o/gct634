import torch
import models
from hparams import hparams

import os

def device_name(device):
    if device == 0:
        device_name = 'CPU'
    else:
        device_name = 'GPU:' + str(device - 1)
    return device_name

best_acc = 0
avg = 0

for i in range(8):
    checkpoint_file = hparams.model_path + '/device_' + device_name(i+1)
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        print(device_name(i+1))
        print("Test Accuracy: %.2f%%" % (100*checkpoint['test_acc']))

        if checkpoint['test_acc'] > best_acc:
            best_acc = checkpoint['test_acc']
        avg += checkpoint['test_acc']

avg /= 8

print("============================")
print("Best Accuracy: %.2f%%" % (100*best_acc))
print("Average: %.2f%%" % (100*avg))

