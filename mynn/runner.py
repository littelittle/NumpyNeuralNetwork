import numpy as np
import os
from tqdm import tqdm
import time
from dataset.data_augmentation import augment_image

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
        per_epoch = kwargs.get("per_epoch", False)
        data_augmentation = kwargs.get("data_augmentation", False)
        # import ipdb; ipdb.set_trace()
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            if os.path.exists(os.path.join(save_dir, 'best_model.pickle')):
                print("load the best model from the last training")
                self.model.load_model(os.path.join(save_dir, 'best_model.pickle'))

        best_score = self.model.dev_score if hasattr(self.model, 'dev_score') else 0.0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))
            X = X[idx]
            y = y[idx]

            for iteration in tqdm(range(int(X.shape[0] / self.batch_size) + 1), desc=f"iter in the {epoch+1} epoch"):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]
                
                if data_augmentation:
                    train_X = train_X.reshape(train_X.shape[0], 28, 28)
                    train_X = np.array([augment_image(img) for img in train_X])
                    train_X = train_X.reshape(train_X.shape[0], -1)

                # iter_start_time = time.time()
                logits = self.model(train_X)
                # forward_finish_time = time.time()
                # print(f"forward_cost is {forward_finish_time-iter_start_time}")
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()
                # backward_finish_time = time.time()
                # print(f"backward_cost is {backward_finish_time-forward_finish_time}")

                self.optimizer.step()
                # print(f"optimize time is {time.time()-backward_finish_time}")
                
                # import ipdb; ipdb.set_trace()

                if (iteration) % log_iters == 0 and iteration != 0:
                    dev_score, dev_loss = self.evaluate(dev_set)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

            # for every epoch, test if it is need to decrease the learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            if per_epoch:
                dev_score, dev_loss = self.evaluate(dev_set)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)
                print(f"epoch: {epoch}, iteration: {iteration}")
                print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                print(f"[Dev] loss: {dev_loss}, score: {dev_score}")            

            if dev_score > best_score:
                if save_dir is not None:
                    print(f"save the best model to {save_dir}")
                    save_path = os.path.join(save_dir, 'best_model.pickle')
                    self.save_model(save_path, dev_score)
                    print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        with self.model.eval(verbose=True):
            logits = self.model(X)
            loss = self.loss_fn(logits, y)
            score = self.metric(logits, y)
        return score, loss
    
    def save_model(self, save_path, dev_score):
        self.model.save_model(save_path, dev_score)