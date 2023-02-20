# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:55:14 2022

@author: ZESHAN KAHN
"""

Y1=[label_map16.index(y) for y in train_Y]
train_Ys=np.zeros((len(Y1),len(label_map16)),np.uint8)
for i in range(len(Y1)):
    train_Ys[i,int(Y1[i])]=1 
    
Y1=[label_map16.index(y) for y in test_Y]
test_Ys=np.zeros((len(Y1),len(label_map16)),np.uint8)
for i in range(len(Y1)):
    test_Ys[i,int(Y1[i])]=1 

from tensorflow.python.platform import tf_logging as logging

class ReduceLRBacktrack(ReduceLROnPlateau):
    def __init__(self, best_path, *args, **kwargs):
        super(ReduceLRBacktrack, self).__init__(*args, **kwargs)
        self.best_path = best_path

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                             self.monitor, ','.join(list(logs.keys())))
        if not self.monitor_op(current, self.best): # not new best
            if not self.in_cooldown(): # and we're not in cooldown
                if self.wait+1 >= self.patience: # going to reduce lr
                    # load best model so far
                    print("Backtracking to best model before reducting LR")
                    self.model.load_weights(self.best_path)

        super().on_epoch_end(epoch, logs) # actually reduce LR
model_checkpoint_path = "/kaggle/working/updatingLR_best.h5"
c1 = ModelCheckpoint(model_checkpoint_path,save_best_only=True,monitor='loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-7)
c2 = ReduceLRBacktrack(best_path=model_checkpoint_path, monitor='loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-7)

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

from keras import Sequential, backend as K, optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adagrad, Adam, Nadam

model = Sequential()
model.add(Dense(64, input_dim=train_X.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='softmax'))

opt = Nadam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


history = model.fit(train_X, train_Ys, epochs=150, batch_size=32,verbose=0,shuffle=True,callbacks=[c1,c2])

pred_Ys = model.predict(test_X)
pred_Y=np.zeros((pred_Ys.shape[0]),np.uint8)
for i in range(len(pred_Y)):
    pred_Y[i]=np.argmax(pred_Ys[i,:])

actual_Y=np.zeros((test_Ys.shape[0]),np.uint8)
for i in range(len(test_Y)):
    actual_Y[i]=np.argmax(test_Ys[i,:])

a = accuracy_score(pred_Y,actual_Y)
print('Accuracy is:', a)