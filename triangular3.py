'''
Code from https://gist.github.com/jeremyjordan/5a222e04bb78c242f5763ad40626c452
'''


from keras.callbacks import Callback
import keras.backend as K
import numpy as np


class Triangular3Scheduler(Callback):
    '''Triangular3 learning rate scheduler with cycles of steeper
    increasing and slowlier decreasing learning rates.

    # Usage
        ```python
            schedule = Triangular3Scheduler(min_lr=1e-5,
                                            max_lr=1e-2,
                                            steps_per_epoch=np.ceil(epoch_size/batch_size),
                                            lr_decay=0.9,
                                            cycle_length=5,
                                            mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range.
        max_lr: The upper bound of the learning rate range.
        steps_per_epoch: Number of mini-batches in the dataset.
                         Calculated as `np.ceil(epoch_size/batch_size)`.
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle,
                  set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        upward_ratio: How much in a cycle to increase learning rate
                      (from min to max). Note learning rate would be
                      decreased for the remaining of the cycle.
        mult_factor: Scale epochs_to_restart after each full cycle
                     completion.

    # References
        Blog post: jeremyjordan.me/nn-learning-rate
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 upward_ratio=0.1
                 mult_factor=1):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.upward_ratio = upward_ratio
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        if fraction_to_restart <= self.upward_ratio:
            # learning rate is increasing
            lr = self.min_lr + (self.max_lr - self.min_lr) * (fraction_to_restart / self.upward_ratio)
        else:
            # learning rate is decreasing
            lr = self.min_lr + (self.max_lr - self.min_lr) * ((1.0 - fraction_to_restart) / (1.0 - self.upward_ratio))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)
