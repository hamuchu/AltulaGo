# coding:utf-8
# TODO: batchでまんべんなくデータが含まれるようにする。
# sgf読み込みで新しい棋譜になったら盤面を初期化する
import pyximport

pyximport.install()
import sgf  # Please install "sgf 0.5". You can install it by using the command of "pip install sgf".
import sys
import re
import os  # osモジュールのインポート
from game import Player
from game import State
from search import MontecarloSearch, DeepLearningSearch
from go import Go
from input_plane import MakeInputPlane
import tensorflow as tf
import math
from go import GoVariable
from go import GoStateObject
from numpy import *

from keras.utils import Sequence
from pathlib import Path
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras
class ForwardPropNetwork(GoVariable):
    character_list = [chr(i) for i in range(97, 97 + 26)]

    def __init__(self,sess):
        self.make_input = MakeInputPlane()

        self.model=self.Network()
        self.model.load_weights('../Network/weights06.hdf5')

        sys.stderr.write(str("Network Model builded"))


    def Network(self):
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


    def search_deep_learning(self,go_state_obj, current_player):
        xTrain = np.array([self.make_input.generate_input(go_state_obj, current_player)])
        #output_array = sess.run(self.y_conv, feed_dict={self.x_input: xTrain})
        output_array=self.model.predict(xTrain)
        print(output_array)
        #two_dimensional_array = tf.reshape(output_array, [-1, 19, 19, 1])
        return output_array

    letter_coords = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

    def coord_to_str(self, row, col):
        return self.letter_coords[row] + str(col + 1)  # 数字をcharacterに変換
    def gtp_io(self, sess):
        players = [Player(0, 'human'), Player(1, 'human')]
        players[0].next_player = players[1]
        players[1].next_player = players[0]
        player = players[0]
        state = Go()
        go_state_obj = GoStateObject()
        # search_algorithm = SimpleSearch()
        search_algorithm = DeepLearningSearch()
        """ Main loop which communicates to gogui via GTP"""
        known_commands = ['boardsize', 'clear_board', 'komi', 'play', 'genmove',
                          'final_score', 'quit', 'name', 'version', 'known_command',
                          'list_commands', 'protocol_version', 'gogui-analyze_commands']
        analyze_commands = ["gfx/Predict Final Ownership/predict_ownership",
                            "none/Load New SGF/loadsgf"]
        # print("starting main.py: loading %s" %sgf_file,file=sys.stderr)
        # output_file = open("output.txt", "wb")
        # output_file.write("intializing\n")
        sys.stderr.write(str("19路盤専用！"))

        while True:
            try:
                line = raw_input().strip()
                # print line
                # output_file.write(line + "\n")
            except EOFError:
                # output_file.write('Breaking!!\n')
                break
            if line == '':
                continue
            command = [s.lower() for s in line.split()]
            # print command
            if re.match('\d+', command[0]):
                command = command[1:]

            ret = ''
            if command[0] == "clear_board":
                state = Go()
                go_state_obj = GoStateObject()
                ret = "= \n\n"
            elif command[0] == "komi":
                ret = "= \n\n"
            elif command[0] == "play":  # playにバグあり "play w C19"が渡された時はx座標,yの順番
                if command[2] == "pass":
                    sys.stderr.write(str(state.print_board(go_state_obj)))
                else:
                    if command[1][0] == "b":
                        player = players[0]
                    elif command[1][0] == "w":
                        player = players[1]

                    x = self.letter_coords.index(command[2][0].upper())  # リストは0からスタート
                    y = 18 - (int(command[2][1:]) - 1)

                    go_state_obj = state.move(go_state_obj, player, (x, y))
                    sys.stderr.write(str(player.player_id))
                    # print go_state_obj._board
                    # print state.print_board(go_state_obj)

                ret = "= \n\n"

            elif command[0] == "genmove":
                if command[1] == "b":
                    player = players[0]
                elif command[1] == "w":
                    player = players[1]
                (go_state_obj, move) = search_algorithm.next_move(self,sess, go_state_obj, player)
                # tup=[move[0],move[1]]
                if move == "pass":
                    ret = "pass"
                else:
                    ret = self.coord_to_str(move[0], 18 - move[1])
                    sys.stderr.write(str("move x:" + str(move[0]) + "\n"))
                    sys.stderr.write(str("move: y" + str(move[1]) + "\n"))

                    sys.stderr.write(str(state.print_board(go_state_obj)))

                ret = '= ' + ret + '\n\n'

            elif command[0] == "final_score":
                # print("final_score not implemented", file=sys.stderr)
                ret = "= \n\n"
            elif command[0] == "name":
                ret = '= EsperanzaGo\n\n'
            elif command[0] == "predict_ownership":
                # ownership_prediction = driver.evaluate_current_board()
                # ret = influence_str(ownership_prediction)
                ret = "= \n\n"
            elif command[0] == "version":
                ret = '= 1.0\n\n'
            elif command[0] == "list_commands":
                # ret = '= \n'.join(known_commands)
                ret = "= boardsize\nclear_board\nquit\nprotocol_version\n" + "name\nversion\nlist_commands\nkomi\ngenmove\nplay\n\n"
            elif command[0] == "gogui-analyze_commands":
                ret = '\n'.join(analyze_commands)

            elif command[0] == "known_command":
                ret = 'true' if command[1] in known_commands else 'false'
            elif command[0] == "protocol_version":
                ret = '= 2\n\n'
            elif command[0] == "boardsize":
                ret = '= \n\n'
            elif command[0] == "quit":
                # ret = '= \n\n'
                # print ret
                # print('=%s \n\n' % (cmdid,), end='')
                exit(0)
                break
            else:
                # print 'Warning: Ignoring unknown command'
                pass
            print(ret)
            sys.stdout.flush()
    def test(self, sess):
        players = [Player(0.0, 'human'), Player(1.0, 'human')]
        players[0].next_player = players[1]
        players[1].next_player = players[0]
        rule = Go()

        player = players[0]
        go_state_obj = GoStateObject()

        x_train = [[self.reshape_board(self.make_input.generate_input(go_state_obj, player))]]
        #print sess.run(self.y_conv, feed_dict={self.x_input: x_train})

    def _main(self,sess):
        pass
        #self.test(sess)
        #print "gtp_io"
        #gtp_io(sess)は普段はコメントアウトすること。
        #self.gtp_io(sess)


#if __name__ == "__main__":
    #ForwardPropNetwork(sess)
