import mxnet as mx
import numpy as np

# Declare function to check for win
def check(board):
    wins = [[0 ,1 ,2], [3 ,4 ,5], [6 ,7 ,8], [0 ,3 ,6], [1 ,4 ,7], [2 ,5 ,8], [0 ,4 ,8], [2 ,4 ,6]]
    for i in range(len(wins)):
        if board[wins[i][0] ] ==board[wins[i][1] ] ==board[wins[i][2] ] ==1.:
            return(1)
        elif board[wins[i][0]]==board[wins [i ][1]]==board[wins [i ][2]]==-1.:
            return(1)
    return(0)

# Print a board
def print_board(board):
    symbols = ['O',' ','X']
    board_plus1 = [int(x) + 1 for x in board]
    print(' ' + symbols[board_plus1[0]] + ' | ' + symbols[board_plus1[1]] + ' | ' + symbols[board_plus1[2]])
    print('___________')
    print(' ' + symbols[board_plus1[3]] + ' | ' + symbols[board_plus1[4]] + ' | ' + symbols[board_plus1[5]])
    print('___________')
    print(' ' + symbols[board_plus1[6]] + ' | ' + symbols[board_plus1[7]] + ' | ' + symbols[board_plus1[8]])

def init_weights(shape):

    b=mx.random.normal(0, 1, shape)
    return b

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def model(X, A1, A2, bias1, bias2):
    a1=mx.nd.multiply(X, A1)
    a2=mx.nd.add(a1,bias1)

    layer1 = sigmoid(a2)
    layer2 = mx.nd.add(mx.nd.multiply(layer1, A2), bias2)
    return(layer2)
