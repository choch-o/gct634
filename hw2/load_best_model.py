import torch
import numpy as np

test_acc = []
state = torch.load('./best_models/GPU5_8414_8100/device_GPU:0')
test_acc.append(state['test_acc'])
print(state['test_acc'])

state = torch.load('./best_models/GPU5_8414_8100/device_GPU:1')
test_acc.append(state['test_acc'])
print(state['test_acc'])
state = torch.load('./best_models/GPU5_8414_8100/device_GPU:2')
test_acc.append(state['test_acc'])
print(state['test_acc'])
state = torch.load('./best_models/GPU5_8414_8100/device_GPU:3')
test_acc.append(state['test_acc'])
print(state['test_acc'])
state = torch.load('./best_models/GPU5_8414_8100/device_GPU:4')
test_acc.append(state['test_acc'])
print(state['test_acc'])
state = torch.load('./best_models/GPU5_8414_8100/device_GPU:5')
test_acc.append(state['test_acc'])
print(state['test_acc'])
state = torch.load('./best_models/GPU5_8414_8100/device_GPU:6')
test_acc.append(state['test_acc'])
print(state['test_acc'])
state = torch.load('./best_models/GPU5_8414_8100/device_GPU:7')
test_acc.append(state['test_acc'])
print(state['test_acc'])

print(test_acc)
print(np.std(test_acc))
