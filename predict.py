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
        model.add(Activation('linear'))

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.000001, decay=1e-13)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        return model


train_dir = pathlib.Path('../KifuLarge')
train_datagen = ImageDataGenerator()

model = Network()
model.load_weights('../Network/weights05.hdf5')

model.predict

#int(np.ceil(len(list(train_dir.iterdir())) / 32))







