# coding:utf-8
# TODO: batchでまんべんなくデータが含まれるようにする。
# TODO: オブジェクトだからこのファイルを実行しても何も起きないので、mainから実行する。
# sgf読み込みで新しい棋譜になったら盤面を初期化する
import pyximport; pyximport.install()
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
#from tensorflow.python import control_flow_ops


# デフォルトの文字コードを出力する
# from guppy import hpy
# h = hpy()


class Train(GoVariable):
    character_list = [chr(i) for i in range(97, 97 + 26)]

    def __init__(self):
        # self.train()
        self.train()

    def reshape_board(self, board_array):
        reshaped_boards = []
        for i in xrange(len(board_array)):
            reshaped_boards.append(reshape(board_array[i], 361))
        return reshaped_boards

    def reshape_answer_board(self, board_array):
        return reshape(board_array, 361)

    def invert_board_input(self, board_array):
        for i in xrange(len(board_array)):
            board_array[i] = board_array[i][::-1]
        return board_array

    def invert_board_answer(self, board_array):
        board_array[::-1]
        return board_array

    def rotate90_input(self, board_array):
        for i in xrange(len(board_array)):
            board_array[i] = rot90(board_array[i])
        return board_array

    def rotate90_answer(self, board_array):
        # 90度回転させるために、配列を２次元にした方が良い。input shapeもNone,1,361にする。
        rot90(board_array)
        return board_array

    def make_rotated_train_batch(self,xTrain,yTrain,input_board,answer_board):
        xTrain.append(self.reshape_board(input_board))
        yTrain.append(self.reshape_answer_board(answer_board))
        # print self.reshape_answer_board(answer_board)
        # print self.reshape_answer_board(answer_board)
        input_board2 = self.rotate90_input(input_board)
        answer_board2 = self.rotate90_answer(answer_board)
        xTrain.append(self.reshape_board(input_board2))
        yTrain.append(self.reshape_answer_board(answer_board2))

        input_board3 = self.rotate90_input(input_board2)
        answer_board3 = self.rotate90_answer(answer_board2)
        xTrain.append(self.reshape_board(input_board3))
        yTrain.append(self.reshape_answer_board(answer_board3))

        input_board4 = self.rotate90_input(input_board3)
        answer_board4 = self.rotate90_answer(answer_board3)
        xTrain.append(self.reshape_board(input_board4))
        yTrain.append(self.reshape_answer_board(answer_board4))

        input_board5 = self.invert_board_input(input_board4)
        answer_board5 = self.invert_board_answer(answer_board4)
        xTrain.append(self.reshape_board(input_board5))
        yTrain.append(self.reshape_answer_board(answer_board5))

        input_board6 = self.rotate90_input(input_board5)
        answer_board6 = self.rotate90_answer(answer_board5)
        xTrain.append(self.reshape_board(input_board6))
        yTrain.append(self.reshape_answer_board(answer_board6))

        input_board7 = self.rotate90_input(input_board6)
        answer_board7 = self.rotate90_answer(answer_board6)
        xTrain.append(self.reshape_board(input_board7))
        yTrain.append(self.reshape_answer_board(answer_board7))

        input_board8 = self.rotate90_input(input_board7)
        answer_board8 = self.rotate90_answer(answer_board7)
        xTrain.append(self.reshape_board(input_board8))
        yTrain.append(self.reshape_answer_board(answer_board8))

        return xTrain,yTrain

    def train(self):
        players = [Player(0.0, 'human'), Player(1.0, 'human')]
        players[0].next_player = players[1]
        players[1].next_player = players[0]
        # player = players[0]
        rule = Go()

        files = os.listdir(os.getcwd() + "/../../kifu")
        print("kifu loaded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        init = tf.global_variables_initializer()
        xTrain = []
        yTrain = []

        num = 0
        batch_count_num = 0
        train_count_num = 0
        make_input = MakeInputPlane()
        step = 0
        ckpt_num = 100
        batch_count_sum_all = 0
        character_list = [chr(i) for i in range(97, 97 + 26)]

        for _ in xrange(100):
            print("kifu passed")
            continue_kifu_num = 0
            for file_name in files:
                # print h.heap()
                continue_kifu_num += 1
                if continue_kifu_num < 150:
                     continue

                step += 1
                with open(os.getcwd() + "/../../kifu/" + file_name) as f:
                    try:
                        collection = sgf.parse(f.read())
                        flag = False
                    except:
                        continue
                    try:
                        go_state_obj = GoStateObject()

                        # print "通過"
                        for game in collection:
                            for node in game:
                                if flag == False:
                                    flag = True
                                    continue

                                position = list(node.properties.values())[0]
                                xpos = character_list.index(position[0][0])
                                ypos = character_list.index(position[0][1])

                                pos_tuple = (xpos, ypos)
                                # print xpos,ypos
                                #print(pos_tuple)
                                color=list(node.properties.keys())[0]
                                if color == "B":
                                    current_player = players[0]
                                elif color=='W':
                                    current_player = players[1]
                                # print "move ends"
                                num += 1
                                if num > 90:
                                    input_board = make_input.generate_input(go_state_obj, current_player)
                                    answer_board = make_input.generate_answer(pos_tuple)

                                    xTrain,yTrain = self.make_rotated_train_batch(xTrain,yTrain,input_board,answer_board)

                                    num = 0
                                    batch_count_num += 1

                                # 注意　moveはinputを作成した後にすること。
                                go_state_obj = rule.move(go_state_obj, current_player, pos_tuple)
                                rule.move(go_state_obj, current_player, pos_tuple)

                                if batch_count_num > 50:
                                    np.savez_compressed('./npzkifu/kifu'+str(train_count_num)+'.npz', x=xTrain, y=yTrain)
                                    #train_step.run(feed_dict={x_input: xTrain, y_: yTrain, keep_prob: 0.5})

                                    batch_count_sum_all += 1
                                    batch_count_num = 0
                                    train_count_num += 1
                                    xTrain = []
                                    yTrain = []
                                    print(train_count_num)
                                if train_count_num > 1000:
                                    train_count_num = 0
                                    ckpt_num += 1
                                    print("SAVED!")

                                    saver.save(sess, './Network/policy_legal_move' + str(ckpt_num))
                    except:
                        traceback.print_exc()
                        f.close()
                        pass
