# coding:utf-8
# TODO: batchでまんべんなくデータが含まれるようにする。
# TODO: オブジェクトだからこのファイルを実行しても何も起きないので、mainから実行する。
# sgf読み込みで新しい棋譜になったら盤面を初期化する

import numpy as np
import os

files = os.listdir(os.getcwd() + "/npzkifu")

print(len(files))

xTrain=np.load(os.getcwd() + "/npzkifu/"+str(files[0]))["x"]
print(xTrain)

yTrain=np.load(os.getcwd() + "/npzkifu/"+str(files[0]))["y"]
print(yTrain)

count=0
first=False
file_num=0
for file_name in files[1:]:
    count+=1
    file_num+=1
    if first:
        xTrain = np.load(os.getcwd() + "/npzkifu/" + str(files[0]))["x"]
        yTrain = np.load(os.getcwd() + "/npzkifu/" + str(files[0]))["y"]
        first=False
        continue
    board_batch_array = np.load(os.getcwd() + "/npzkifu/"+str(file_name))
    xTrain=np.append(xTrain,board_batch_array["x"],axis=0)
    yTrain = np.append(yTrain, board_batch_array["y"], axis=0)

    if count >= 500:
        print(file_num)
        print(count)
        count=0
        first=True
        np.savez_compressed('./KifuLarge/KifuLargeFile'+str(file_num)+'.npz', x=xTrain, y=yTrain)
