# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
from dataset.data_augmentation import augment_image

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set, so there is only 50000 left for training.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()


def train_model():
        # specify the child saving path
        save_path = r'./saved_models/base_model.pickle'

        linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4]) 
        conv_model = nn.models.Model_CNN([(1, 6, 5), (2,), (6, 16, 5), (2,),("reshape"),(16*5*5, 120),(120, 84), (84, 10)], "ReLU", [1e-4, 1e-4, 1e-4, 1e-4, 1e-4])

        model = conv_model

        SGD_optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
        Momentum_optimizer = nn.optimizer.MomentGD(init_lr=0.001, model=model, mu=0.9) # 1/(1-mu) of SGD
        optimizer = Momentum_optimizer

        scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[1, 2, 3], gamma=0.5)
        loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max()+1)

        runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

        runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=20, log_iters=100, save_dir=r'./best_models_with_padding', data_augmentation=True)

        _, axes = plt.subplots(1, 2)
        axes.reshape(-1)
        _.set_tight_layout(1)
        plot(runner, axes)

        plt.show()

def test_MLP():
        dev_scores = []
        for i in [300, 600, 1200]:
                model = nn.models.Model_MLP([train_imgs.shape[-1], i, 10], 'ReLU', [1e-4, 1e-4])
                optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
                scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[1, 2, 3], gamma=0.5)
                loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max()+1)
                runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
                runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=3, log_iters=100000, save_dir=None, per_epoch=True)
                dev_scores.append(runner.dev_scores)

        # plot the dev_scores in one figure
        plt.figure()
        plt.plot(dev_scores[0], label='nhidden=300')
        plt.plot(dev_scores[1], label='nhidden=600')
        plt.plot(dev_scores[2], label='nhidden=1200')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('MLP with different hidden layer size')
        plt.show()

def test_dropout():
        dev_scores = []
        for dropout in [0.8, 0.5, False]:
                model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4], dropout=dropout)
                optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
                scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[1, 2, 3], gamma=0.5)
                loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max()+1)
                runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
                runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=3, log_iters=100000, save_dir=None, per_epoch=True)
                dev_scores.append(runner.dev_scores)

        # plot the dev_scores in one figure
        plt.figure()
        plt.plot(dev_scores[0], label='dropout=0.8')
        plt.plot(dev_scores[1], label='dropout=0.5')
        plt.plot(dev_scores[2], label='dropout=0.0')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('MLP with different dropout rate')
        plt.show()

if __name__ == '__main__':
        train_model()
        # test_MLP()
        # test_dropout()