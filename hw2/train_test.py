'''
train_test.py

A file for training model for genre classification.
Please check the device in hparams.py before you run this code.
'''
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import data_manager
import models
from hparams import hparams
import numpy as np

import sys

best_acc = 0
# Wrapper class to run PyTorch model
class Runner(object):
    def __init__(self, hparams):
        self.model = models.Conv_2D(hparams)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cpu")

        if hparams.device > 0:
            torch.cuda.set_device(hparams.device - 1)
            self.model.cuda(hparams.device - 1)
            self.criterion.cuda(hparams.device - 1)
            self.device = torch.device("cuda:" + str(hparams.device - 1))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=hparams.learning_rate, momentum=hparams.momentum, nesterov=True)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=hparams.factor, patience=hparams.patience, verbose=True)
        self.learning_rate = hparams.learning_rate
        self.stopping_rate = hparams.stopping_rate

    # Accuracy function works like loss function in PyTorch
    def accuracy(self, source, target, mode='train'):
        if mode == 'test':
            source = source.long().cpu()
        else:
            source = source.max(1)[1].long().cpu()
        target = target.cpu()
        correct = (source == target).sum().item()

        return correct/float(source.size(0))

    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, mode='train', eval_mode='valid'):
        self.model.train() if mode is 'train' else self.model.eval()

        epoch_loss = 0
        epoch_acc = 0
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            if (eval_mode == 'test') or (eval_mode == 'best'):
                x_slices = [x[:, i:i+hparams.feature_length] for i in range(0, x.size(1), hparams.feature_length)]
                predictions = []
                for x_slice in x_slices:
                    predictions.append(self.model(x_slice).max(1)[1].long().cpu())

                prediction = torch.zeros(x.size(0))
                for b in range(x.size(0)):
                    prediction_count = torch.zeros(len(hparams.genres))
                    for p in predictions:
                        prediction_count[p[b]] += 1
                    prediction[b] = np.argmax(prediction_count)

                acc = self.accuracy(prediction, y, mode='test')
            else:
                prediction = self.model(x)
                loss = self.criterion(prediction, y)
                acc = self.accuracy(prediction, y)

            if mode is 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if eval_mode != 'test':
                epoch_loss += prediction.size(0)*loss.item()
            epoch_acc += prediction.size(0)*acc

        epoch_loss = epoch_loss/len(dataloader.dataset)
        epoch_acc = epoch_acc/len(dataloader.dataset)

        if mode is 'eval':
            if eval_mode is 'test':
                self.save_checkpoint(epoch_acc, hparams.model_path + '/device_' + device_name(hparams.device))
        return epoch_loss, epoch_acc

    # Early stopping function for given validation loss
    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate

        return stop

    def save_checkpoint(self, test_acc, checkpoint_path):
        state = {}
        state['test_acc'] = test_acc
        state['model_state'] = self.model.state_dict()
        state['optimizer_state'] = self.optimizer.state_dict()

        torch.save(state, checkpoint_path)

def device_name(device):
    if device == 0:
        device_name = 'CPU'
    else:
        device_name = 'GPU:' + str(device - 1)

    return device_name

def main():
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(hparams)
    runner = Runner(hparams)

    print('Training on ' + device_name(hparams.device))

    for epoch in range(hparams.num_epochs):
        train_loss, train_acc = runner.run(train_loader, 'train', 'train')
        valid_loss, valid_acc = runner.run(valid_loader, 'eval', 'valid')

        print("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f] [Valid Loss: %.4f] [Valid Acc: %.4f]" %
              (epoch + 1, hparams.num_epochs, train_loss, train_acc, valid_loss, valid_acc))
        if runner.early_stop(valid_loss, epoch + 1):
            break
    test_loss, test_acc = runner.run(test_loader, 'eval', 'test')
    print("Training Finished")
    print("Test Accuracy: %.2f%%" % (100*test_acc))

if __name__ == '__main__':
    main()
