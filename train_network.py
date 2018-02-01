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

from keras.models                      import Sequential, Model
from keras.layers                      import Dense, Input, MaxPool2D, concatenate
from keras.layers.core                 import Activation, Flatten, Dropout
from keras.layers.pooling              import AvgPool2D
from keras.optimizers                  import Adam
from keras.utils                       import plot_model
from keras.callbacks                   import TensorBoard
from keras.layers.normalization        import BatchNormalization
from keras.layers.convolutional        import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras_contrib.layers              import normalization, advanced_activations

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers import Conv2D, MaxPooling2D
import keras
import pickle
#from tensorflow.python import control_flow_ops


# デフォルトの文字コードを出力する
# from guppy import hpy
# h = hpy()
from keras.utils import Sequence
from pathlib import Path
import pandas
import numpy as np
from keras.utils import np_utils


class CSVSequence(Sequence):
    def __init__(self, kind, length):
        # コンストラクタ
        self.kind = kind
        self.length = length
        self.data_file_path = str(Path(download_path) / self.kind / "splited" / "split_data_{0:05d}.csv")

    def __getitem__(self, idx):
        # データの取得実装
        data = pandas.read_csv(self.data_file_path.format(idx), encoding="utf-8")
        data = data.fillna(0)

        # 訓練データと教師データに分ける
        x_rows, y_rows = get_data(data)

        # ラベルデータのカテゴリカル変数化
        Y = np_utils.to_categorical(y_rows, nb_classes)
        X = np.array(x_rows.values)

        return X, Y

    def __len__(self):
        # 全データの長さ
        return self.length

    def on_epoch_end(self):
        # epoch終了時の処理
        pass



class Train(GoVariable):
    def __init__(self):
        self.train()
    def train(self):
        file_lists = os.listdir("../KifuLarge/")
        print(file_lists)

        model=self.Network()
        for i in range(0,100):
            for npzfile in file_lists:
                xTrain = np.load("../KifuLarge/" + str(npzfile))["x"]
                yTrain = np.load("../KifuLarge/" + str(npzfile))["y"]
                print(xTrain.shape)
                xTrain=np.reshape(xTrain,(204000,5,19,19))
                model.fit(x=xTrain, y=yTrain, shuffle=True,batch_size=100,epochs=2)
            model.save('../Network/model'+str(i)+'.hdf5')

    def Network(self):
        model = Sequential()
        #model.add(Reshape((5,19,19)))
        model.add(Conv2D(256, (4, 4), padding='same',input_shape=(5,19,19)))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (2, 2), padding='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(256, (2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (2, 2), padding='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(256, (2, 2), padding='same'))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(600))
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
    def train_generator(self):
        from pathlib import Path
        import multiprocessing

        # csvダウンロード先パス
        download_path = "/data"

        # 同時実行プロセス数
        process_count = multiprocessing.cpu_count() - 1


        base_dir = Path(download_path)

        # 訓練データ
        train_data_dir = base_dir / "log_quest" / "splited"
        train_data_file_list = list(train_data_dir.glob('split_data_*.csv'))
        train_data_file_list = train_data_file_list

        #検証用データ
        val_data_dir = base_dir / "log_quest_validate" / "splited"
        val_data_file_list = list(val_data_dir.glob('split_data_*.csv'))
        val_data_file_list = val_data_file_list

        history = model.fit_generator(CSVSequence("log_quest", len(train_data_file_list)),
                        steps_per_epoch=len(train_data_file_list), epochs=1, max_queue_size=process_count * 10,
                        validation_data=CSVSequence("log_quest_validate", len(val_data_file_list)), validation_steps=len(val_data_file_list),
                        use_multiprocessing=True, workers=process_count)
