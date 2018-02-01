# coding:utf-8
# TODO: batchでまんべんなくデータが含まれるようにする。
# TODO: オブジェクトだからこのファイルを実行しても何も起きないので、mainから実行する。
# sgf読み込みで新しい棋譜になったら盤面を初期化する
import pyximport;

pyximport.install()
import sgf  # Please install "sgf 0.5". You can install it by using the command of "pip install sgf".
import re
import os  # osモジュールのインポート
from game import Player
from game import State
from search import MontecarloSearch
from go import Go
from input_plane import MakeInputPlane
import tensorflow as tf
import math
from go import GoVariable
from go import GoStateObject
from numpy import *
import traceback
# パスはどうする？forwardした結果一番良い答えがパスかもしれない
import sys
import datetime
import numpy as np
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Input, MaxPool2D, concatenate
from keras.layers.core import Activation, Flatten, Dropout
from keras.layers.pooling import AvgPool2D
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras_contrib.layers import normalization, advanced_activations

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
import keras
import pickle
import pathlib
# from tensorflow.python import control_flow_ops


# デフォルトの文字コードを出力する
# from guppy import hpy
# h = hpy()
from keras.utils import Sequence
from pathlib import Path
import numpy as np
from keras.utils import np_utils


def Network():
        model = Sequential()
        # model.add(Reshape((5,19,19)))
        model.add(Conv2D(128, (4, 4), padding='same', input_shape=(5, 19, 19)))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (2, 2), padding='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(128, (2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (2, 2), padding='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(128, (2, 2), padding='same'))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(361))
        model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        return model





'''
    file_lists = os.listdir("../KifuLarge/")
    print(file_lists)

    model = self.Network()
    for i in range(0, 100):
        for npzfile in file_lists:
            xTrain = np.load("../KifuLarge/" + str(npzfile))["x"]
            yTrain = np.load("../KifuLarge/" + str(npzfile))["y"]
            print(xTrain.shape)
            xTrain = np.reshape(xTrain, (204000, 5, 19, 19))
            model.fit(x=xTrain, y=yTrain, shuffle=True, batch_size=100, epochs=2)
        model.save('../Network/model' + str(i) + '.hdf5')
'''

class ImageDataGenerator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.images = None
        self.labels = None

    def flow_from_directory(self):

        while True:
            # ディレクトリから画像のパスを取り出す
            for path in pathlib.Path(train_dir).iterdir():
                print(path)
                if path == "../KifuLarge/KifuLargeFile3000.npz":
                    print("load is done. successfully loaded")
                    exit(0)
                # 画像を読み込みRGBへの変換、Numpyへの変換を行い、配列(self.iamges)に格納

                self.images = np.reshape(np.load(path)["x"],(204000,5,19,19))
                self.labels = np.load(path)["y"]

                # ここまでを繰り返し行い、batch_sizeの数だけ配列(self.iamges, self.labels)に格納
                # batch_sizeの数だけ格納されたら、戻り値として返し、配列(self.iamges, self.labels)を空にする
                inputs = np.asarray(self.images, dtype=np.float32)
                targets = np.asarray(self.labels, dtype=np.float32)

                for i in range(0,int(inputs.size/50)):
                    if int(inputs.size/50)-1==i:
                        yield inputs[i:], targets[i:]
                    else:
                        yield inputs[i:i+50], targets[i:i+50]



train_dir = pathlib.Path('../KifuLarge')
train_datagen = ImageDataGenerator()


model = Network()
model.fit_generator(
    generator=train_datagen.flow_from_directory(),
    steps_per_epoch=500000,epochs=1,verbose=1)

model.save_weights('../Network/model.hdf5')


#int(np.ceil(len(list(train_dir.iterdir())) / 32))







