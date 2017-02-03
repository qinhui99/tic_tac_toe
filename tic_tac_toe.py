#encoding=utf-8
# I see the funny tic-tac-toe codes from https://github.com/nfmcclure/tensorflow_cookbook.
# The original code is in https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks/08_Learning_Tic_Tac_Toe.
#
# I  rewrite these codes in mxnet API. Although it is still naive, it works.


import mxnet as mx
import numpy as np
import csv
import random
import comm_func
import logging

# X = 1
# O = -1
# empty = 0
# response on 1-9 grid for placement of next '1'


# For example, the 'test_board' is:
#
#   O  |  -  |  -
# -----------------
#   X  |  O  |  O
# -----------------
#   -  |  -  |  X
#

response = 6
symmetry = ['rotate180', 'rotate90', 'rotate270', 'flip_v', 'flip_h']

## Given a board, a response, and a transformation, get the new board+response
def get_symmetry(board, response, transformation):
    '''
    :param board: list of integers 9 long:
     opposing mark = -1
     friendly mark = 1
     empty space = 0
    :param transformation: one of five transformations on a board:
     'rotate180', 'rotate90', 'rotate270', 'flip_v', 'flip_h'
    :return: tuple: (new_board, new_response)
    '''
    if transformation == 'rotate180':
        new_response = 8 - response
        return(board[::-1], new_response)
    elif transformation == 'rotate90':
        new_response = [6, 3, 0, 7, 4, 1, 8, 5, 2].index(response)
        tuple_board = list(zip(*[board[6:9], board[3:6], board[0:3]]))
        return([value for item in tuple_board for value in item], new_response)
    elif transformation == 'rotate270':
        new_response = [2, 5, 8, 1, 4, 7, 0, 3, 6].index(response)
        tuple_board = list(zip(*[board[0:3], board[3:6], board[6:9]]))[::-1]
        return([value for item in tuple_board for value in item], new_response)
    elif transformation == 'flip_v':
        new_response = [6, 7, 8, 3, 4, 5, 0, 1, 2].index(response)
        return(board[6:9] +  board[3:6] + board[0:3], new_response)
    elif transformation == 'flip_h':  # flip_h = rotate180, then flip_v
        new_response = [2, 1, 0, 5, 4, 3, 8, 7, 6].index(response)
        new_board = board[::-1]
        return(new_board[6:9] +  new_board[3:6] + new_board[0:3], new_response)
    else:
        raise ValueError('Method not implmented.')

## Read in board move csv file
def get_moves_from_csv(csv_file):
    '''
    :param csv_file: csv file location containing the boards w/ responses
    :return: moves: list of moves with index of best response
    '''
    moves = []
    with open(csv_file, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            moves.append(([int(x) for x in row[0:9]],int(row[9])))
    return(moves)

# Get random board with optimal move
def get_rand_move(moves, n=1, rand_transforms=2):
    '''
    :param moves: list of the boards w/responses
    :param n: how many board positions with responses to return in a list form
    :param rand_transforms: how many random transforms performed on each
    :return: (board, response), board is a list of 9 integers, response is 1 int
    '''
    (board, response) = random.choice(moves)
    possible_transforms = ['rotate90', 'rotate180', 'rotate270', 'flip_v', 'flip_h']
    for i in range(rand_transforms):
        random_transform = random.choice(possible_transforms)
        (board, response) = get_symmetry(board, response, random_transform)
    return(board, response)

# Get list of optimal moves w/ responses
moves = get_moves_from_csv('base_tic_tac_toe_moves.csv')


test_board = [-1, 0, 0, 1, -1, -1, 0, 0, 1]

# Create a train set:
train_length = 500

train_data=[]
target_data=[]
for t in range(train_length):
    b,r=get_rand_move(moves)
    # print (r)
    if b!=test_board:
        train_data.append(b)
        target_data.append(r)


# Learning Optimal Tic-Tac-Toe Moves via a Neural Network
#---------------------------------------
#
# We will build a two-hidden layer neural network
#  to predict the optimal response given a set
#  of tic-tac-toe boards.
x_sym = mx.symbol.Variable('data')
y_sym = mx.symbol.Variable('softmax_label')
fc1 = mx.sym.FullyConnected(data=x_sym, name='fc1', num_hidden=100)
act1 = mx.symbol.Activation(data = fc1, name='act1', act_type="relu")
fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=1)
net = mx.sym.LinearRegressionOutput(data=fc2,label=y_sym, name='loss')

n_epochs = 100
learning_rate = 0.025
batch_size =50
# we can not use the mx.mod.Module. Because it will report errors:
#/matrix_op-inl.h:712: Check failed: param.begin[i] < shp[i] && param.end[i] <= shp[i] && param.begin[i] < param.end[i]
# mod = mx.mod.Module(symbol=net,
#                     context=mx.cpu(),
#                     )
#

# mod.fit(train_iter,
#         optimizer='adam',
#         optimizer_params={'learning_rate':learning_rate},
#         eval_metric=eval_metrics,
#         eval_data=train_iter,
#         num_epoch=n_epochs
#         )
mod = mx.model.FeedForward(
    ctx=mx.cpu(), symbol=net, num_epoch=n_epochs,
    learning_rate=learning_rate,
    optimizer='adam'
)
logging.basicConfig(level=logging.INFO)

# Build iterator
slice_index=train_length-350
# slice_index=train_length
train_iter = mx.io.NDArrayIter(data=mx.ndarray.array(train_data[:slice_index]), label=mx.ndarray.array(target_data[:slice_index]),
                               batch_size=batch_size, shuffle=True)
eval_iter = mx.io.NDArrayIter(data=mx.ndarray.array(train_data[slice_index:]), label=mx.ndarray.array(target_data[slice_index:]),
                               batch_size=batch_size, shuffle=True)

eval_metrics = ['mse']
eval_metrics.append('rmse')

mod.fit(train_iter,
        # optimizer='adam',
        # optimizer_params={'learning_rate':learning_rate},
        eval_metric=eval_metrics,
        eval_data=train_iter,
        # arg_params={"A1":a1,"BIAS1":bias1},
        # num_epoch=n_epochs
        )


print('Finished training')

test_boards = [test_board]
r= mx.io.NDArrayIter(data=mx.ndarray.array(test_boards),batch_size=1, shuffle=False)
result=mod.predict(r, num_batch=1)
print ("prediction=",result)
# mod.save_params('my_model')
# Let's play against our model
game_trackers=[]
game_tracker = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
win_logical = False
num_moves = 0
while not win_logical:
    player_index = input('Input index of your move (0-8): ')
    num_moves += 1
    # Add player move to game
    game_tracker[int(player_index)] = 1.
    print (game_tracker)
    # Get model's move by first getting all the logits for each index
    if len(game_trackers)>0:
        game_trackers=[]
    #Because we need a matrix like this:[[0., 0., 0., 0., 0., 0., 0., 0., 0.]]
    game_trackers.append(game_tracker)

    # Now find allowed moves (where game tracker values = 0.0)
    allowed_moves = [ix for ix, x in enumerate(game_tracker) if x == 0.0]
    r1= mx.io.NDArrayIter(data=mx.ndarray.array(game_trackers),
                          batch_size=1, shuffle=False)

    result2=mod.predict(r1, num_batch=1)
    # print ("prediction=",result2)
    if result2<0:
        result2=np.argmin(allowed_moves)
    # potential_moves = np.floor(result2)
    potential_moves = np.round(result2)
    # print ("potential_moves",potential_moves)


    # Now check for win or too many moves
    if len(allowed_moves)==0:
        print('Game Over!')
        win_logical = True

    else:
        # print ("allowed_moves",allowed_moves)
        # Find best move by taking argmax of logits if they are in allowed moves
        if potential_moves in (allowed_moves):
            model_move =potential_moves
        else:
            # model_move = allowed_moves[np.argmax(allowed_moves)]
            temp_moves=allowed_moves-potential_moves
            model_move = allowed_moves[np.argmin(temp_moves)]
            # print ("temp model_move",model_move)

        # print ("model_move",model_move)
        game_trackers=[]
        # Add model move to game
        game_tracker[int(model_move)] = -1.
        game_trackers.append(game_tracker)
        # print ("Model has move and a= ", game_trackers)
        print('Model has moved')
        # Now check for win or too many moves
        if comm_func.check(game_tracker) == 1:
            print('Game Over!')
            win_logical = True
    comm_func.print_board(game_tracker)

