# Step 1: load the Torch library and other utilities
#----------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models,transforms,datasets
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk
from pathlib import Path
import random
import scipy.io
import os
import time
import pdb
from scipy.io import savemat, loadmat
import logging

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")



# Step 2: load the dataset, ie, load human-play of AlphaGo on XXX server
#--------------------------------------------------------------------------------------------------

# alpha go pieces
BLACK_PIECE   =  1
WHITE_PIECE   = -1
NO_MOVE_PIECE =  0
NUMBER_OF_MOVES = 19*19


class SingleBoard(object):

    def __init__(self, input_board, move_row, move_col, output_board, move_piece):
        self.input_board  = input_board
        self.output_board = output_board
        self.move_row     = move_row
        self.move_col     = move_col
        self.move_piece   = move_piece
        self.board_height = len(input_board)
        #pdb.set_trace()
        self.board_width  = len(input_board[0])
    
    def get_input_board(self):
        return self.input_board

    def get_output_board(self):
        return self.output_board

    def get_label(self):
        return self.move_row*self.board_width + self.move_col # label = i*BOARD_WIDTH + j

class GoBoardDataset(object):

    def __init__(self, all_boards, all_labels, board_width=19, board_height=19):
        self.board_width    = board_width
        self.board_height   = board_height
        self.all_boards     = all_boards
        self.all_labels     = all_labels # a label is defined by the position that uniquely identifies the position (i,j) --> i*19 + j
        self.size           = all_boards.shape[0]

    def __getitem__(self, index):
        return (self.all_boards[index], self.all_labels[index])

    def __len__(self):
        return self.size




def parse_a_board(cur_line):

    last_index      = cur_line.rfind(',')
    board           = cur_line[1:last_index-1] # remove 3 items: 1) outer list bracket '[', 2) outer list bracket ']', and 3) the last comma ','
    board_splitted  = board.split('], ')

    final_board     = []
    for j in range(len(board_splitted)):
        cur_board_row = board_splitted[j]

        if j == len(board_splitted)-1:
            cur_board_row = cur_board_row[:-1]

        cur_board_row = cur_board_row[1:]
        splitted_list = cur_board_row.split(', ')
        cur_row_list = []
        for k in range(len(splitted_list)):
            cur_row_list.append(int(splitted_list[k]))
        final_board.append(cur_row_list)

    return final_board


def parse_a_move(move_str):
    splitted_list   = move_str.split(' ')
    move_col        = int(splitted_list[0])
    move_row        = int(splitted_list[1]) - 1 # there was typo maybe in the code that produces a row index starting at 1 instead of 0
    move_piece      = int(splitted_list[2].rstrip('\n'))
    return move_col, move_row, move_piece



def read_board_data_from_file(file_name):
    # 19x19 board along with moves made by humans in real game (downloaded from the online Go-game-server)
    
    empty_row   = [0]*19
    empty_board = [empty_row]*19
    with open(file_name, 'r') as f:
        all_lines = f.readlines()

        all_board_list  = []
        all_label_list  = []

        i = 0
        while i < len(all_lines):

            #print("processing ", all_lines[i])
            
            
            if all_lines[i].rstrip('\n') == 'NewGame':
                last_board      = empty_board
                # increment the counter to read the next line
                i = i + 1

            else:
                
                # first line contains 2d list of board
                # ------------ create a 2d-list of numbers after parsing the elements ------
                cur_line        = all_lines[i]
                final_board     = parse_a_board(cur_line)
                # increment the counter to read the next line
                i = i + 1
                
                '''if (i == 125369):
                    pdb.set_trace()'''


                # second line contains the position (i, j) and the piece_type {1, -1}
                # ------------ save the moves after parsing the elements -------------------
                move_str        = all_lines[i]                
                move_col, move_row, move_piece = parse_a_move(move_str)
                # increment the counter to read the next line
                i = i + 1


                # create a board object
                board_obj       = SingleBoard(last_board, move_row, move_col, final_board, move_piece)
                board_label     = board_obj.get_label()
                
                # create a tensor of size [1x19x19]: as we will be doing Conv2d operation, each input should be a 3D tensor
                '''board_tensor    = torch.tensor(last_board).unsqueeze(dim=0) # by default it is a torch.LongTensor()
                label_tensor    = torch.tensor(board_label).unsqueeze(dim=0)'''
                
                '''board_tensor    = torch.FloatTensor(last_board).unsqueeze(dim=0) #because the network's weights are torch.FloatTensor()
                label_tensor    = torch.FloatTensor(board_label).unsqueeze(dim=0)'''
                
                board_tensor    = torch.FloatTensor(last_board).unsqueeze(dim=0) #because the network's weights are torch.FloatTensor()
                #label_tensor    = torch.tensor(board_label).unsqueeze(dim=0) # this cause issues during the loss computation (dimension mismatch!)
                label_tensor    = torch.tensor(board_label)
                #pdb.set_trace()
                all_board_list.append(board_tensor)
                all_label_list.append(label_tensor)

                # update the information so that in next turn it will be used as input_board
                last_board      = final_board
                
                if i%1 == 0:
                    print(f"processed {i}/{len(all_lines)}")


        #pdb.set_trace()
        # create dataset with tensors for training
        all_board_tensors = torch.stack(all_board_list)
        all_label_tensors = torch.stack(all_label_list)

        return all_board_tensors, all_label_tensors


    

# Step 3: our network
#--------------------------------------------------------------------------------------------------
class AlphaGoPolicyNetwork(nn.Module):

    def __init__(self, board_height, board_width, nf=32, number_of_moves=19*19):
        super(AlphaGoPolicyNetwork, self).__init__()

        self.rows             = board_height
        self.cols             = board_width
        self.input_channel    = 1
        self.conv_window_size = 2

        self.encoder = nn.Sequential(
          nn.Conv2d(self.input_channel,  nf,     self.conv_window_size),
          nn.BatchNorm2d(nf),
          nn.Conv2d(nf, nf*2,   self.conv_window_size),
          nn.BatchNorm2d(nf*2),
          nn.Conv2d(nf*2, nf*4, self.conv_window_size),
          nn.BatchNorm2d(nf*4),
          nn.Conv2d(nf*4, nf*8, self.conv_window_size),
          nn.BatchNorm2d(nf*8)

        )

        self.flatten                    = nn.Flatten()

        # End layers: a series of dense linear layers (almost like an MLP for final classification)
        self.linear_layers = nn.Sequential(
                nn.Linear(256*15*15, 1024),    
                nn.ReLU(),
                nn.Linear(1024, number_of_moves)
            )

    def forward(self, x):
        # forward pass
        output = self.encoder(x)
        #pdb.set_trace()
        output = self.flatten(output)
        output = self.linear_layers(output)

        return output


#--------------------------------------------------------------------------------------------------
class AlphaGoPolicyNetwork_v2(nn.Module):

    def __init__(self, board_height, board_width, nf=32, number_of_moves=19*19):
        super(AlphaGoPolicyNetwork_v2, self).__init__()

        self.rows             = board_height
        self.cols             = board_width
        self.input_channel    = 1
        self.conv_window_size = 2

        self.encoder = nn.Sequential(
          nn.Conv2d(self.input_channel,  nf,     self.conv_window_size),
          nn.BatchNorm2d(nf),
          nn.Conv2d(nf, nf*2,   self.conv_window_size),
          nn.BatchNorm2d(nf*2),
          nn.Conv2d(nf*2, nf*4, self.conv_window_size),
          nn.BatchNorm2d(nf*4),
          nn.Conv2d(nf*4, nf*8, self.conv_window_size),
          nn.BatchNorm2d(nf*8),

          #Added Layers
          nn.Conv2d(nf*8, nf*16,   self.conv_window_size),
          nn.BatchNorm2d(nf*16),
          nn.Conv2d(nf*16, nf*32, self.conv_window_size),
          nn.BatchNorm2d(nf*32),
          nn.Conv2d(nf*32, nf*64, self.conv_window_size),
          nn.BatchNorm2d(nf*64),
          nn.Conv2d(nf*64, nf*128, self.conv_window_size),
          nn.BatchNorm2d(nf*128)
        )

        self.flatten                    = nn.Flatten()

        # End layers: a series of dense linear layers (almost like an MLP for final classification)
        self.linear_layers = nn.Sequential(
                nn.Linear(495616, 4096),
                nn.ReLU(),
                nn.Linear(4096, 1024),
                nn.ReLU(),
                nn.Linear(1024, number_of_moves)
            )

    def forward(self, x):
        # forward pass
        output = self.encoder(x)
        #pdb.set_trace()
        output = self.flatten(output)
        output = self.linear_layers(output)

        return output



# Step 4: Your training and testing functions
#--------------------------------------------------------------------------------------

def train_loop(dataloader, model, loss_fn, optimizer):

    size            = len(dataloader.dataset)
    num_batches     = len(dataloader)

    model.train()                   # set the model to training mode for best practices

    train_loss      = 0
    correct         = 0
    train_pred_all  = []
    train_y_all     = []

    for batch, (X, y) in enumerate(dataloader):
        # compute prediction and loss

        # ----------- putting data into gpu or sticking to cpu ----------
        if torch.cuda.is_available():
          X, y = Variable(X).cuda(), Variable(y).cuda()
        else:
          X, y = Variable(X), Variable(y)
        # -----------                                         ----------

        pred = model(X)
        #pdb.set_trace()
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # compute the accuracy
        pred_prob   = softmax(pred)
        pred_y 			= torch.max(pred_prob, 1)[1]
        train_correct = (pred_y == y).sum()
        correct    += train_correct.data

        train_pred_all.append(pred_y) # save predicted output for the current batch
        train_y_all.append(y)         # save ground truth for the current batch

    #pdb.set_trace()
    train_pred_all = torch.cat(train_pred_all) # need to concatenate batch-wise appended items
    train_y_all = torch.cat(train_y_all)

    train_loss = train_loss/num_batches
    correct    = correct.cpu().numpy()/size
        

    #print('Confusion matrix for training set:\n', confusion_matrix(train_y_all.cpu().data, train_pred_all.cpu().data))
    return train_loss, 100*correct


def test_loop(dataloader, model, loss_fn):

    model.eval()                    # set the model to evaluation mode for best practices

    size                = len(dataloader.dataset)
    num_batches         = len(dataloader)
    test_loss, correct  = 0, 0
    test_pred_all       = []
    test_y_all          = []

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():

      for X, y in dataloader:

        # ----------- putting data into gpu or sticking to cpu ----------
        if torch.cuda.is_available():
          X, y = Variable(X).cuda(), Variable(y).cuda()
        else:
          X, y = Variable(X), Variable(y)
        # -----------                                         ----------

        pred          = model(X)
        test_loss    += loss_fn(pred, y).item()

        # calculate probability and save the outputs for confusion matrix computation
        pred_prob     = softmax(pred)
        pred_y        = torch.max(pred_prob, 1)[1]
        test_correct  = (pred_y == y).sum()
        correct      += test_correct.data

        test_pred_all.append(pred_y) # save predicted output for the current batch
        test_y_all.append(y)         # save ground truth for the current batch


    #pdb.set_trace()
    test_pred_all = torch.cat(test_pred_all)
    test_y_all = torch.cat(test_y_all)

    test_loss = test_loss/num_batches
    correct   = correct.cpu().numpy()/size
    logging.info(f"Test Performance: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    logging.info('Confusion matrix for test set:\n', confusion_matrix(test_y_all.cpu().data, test_pred_all.cpu().data))
    return test_loss, 100*correct, confusion_matrix(test_y_all.cpu().data, test_pred_all.cpu().data)

# Step 5: instantiate the network, prepare the DataLoader and select your optimizer and set the hyper-parameters for learning the model from DataLoader
#------------------------------------------------------------------------------------------------------------------------------

network_type       = 2

if network_type == 1:
    
    policy_network_model = AlphaGoPolicyNetwork(board_height=19, board_width=19, nf=32, number_of_moves=NUMBER_OF_MOVES)
    model_save_path_prefix = 'modelv1_on_data_v2'
    logging.basicConfig(filename=model_save_path_prefix + '/logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"{'AlphaGoPolicyNetwork_v1:':<{20}} {'4 layers of cnn':<{20}}")    

elif network_type == 2:
    
    policy_network_model = AlphaGoPolicyNetwork_v2(board_height=19, board_width=19, nf=32, number_of_moves=NUMBER_OF_MOVES)
    model_save_path_prefix = 'modelv2_on_data_v2'
    logging.basicConfig(filename=model_save_path_prefix + '/logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"{'AlphaGoPolicyNetwork_v2:':<{20}} {'8 layers of cnn':<{20}}")
    
elif network_type == 3:
    
    logging.info(f"{'AlphaGoPolicyNetwork_v3:':<{20}} {'16 layers of cnn':<{20}}")
    logging.info(f"not implemented yet ...")
    
policy_network_model.to(device)
print(policy_network_model)


#training_file_name           = 'data/human_replay_go_test3.txt'
#training_file_name           = 'data/human_replay_go.txt'
#training_file_name           = 'data_v1/TrainData.txt'
training_file_name           = 'data_v2/Train_data.txt'
t0 = time.time()
board_tensors, label_tensors = read_board_data_from_file(training_file_name)
#print(f"Time: it took {time.time()-t0} seconds to read 8 MB input file of human replays of go games")
#print(f"Time: it took {time.time()-t0} seconds to read 189 MB input file of human replays of go games")
print(f"Time: it took {time.time()-t0} seconds to read 8.7 GB input file of human replays of go games")
train_dataset                = GoBoardDataset(board_tensors, label_tensors)



learning_rate      = 1e-4
batch_size_val     = 128
epochs             = 400
loss_fn            = nn.CrossEntropyLoss()
optimizer          = torch.optim.Adam(policy_network_model.parameters(), lr=learning_rate)
softmax            = nn.Softmax(dim=1) # for calculating the probability of the network prediction. It is used in train_loop() and test_loop().

train_dataloader   = DataLoader(train_dataset, batch_size=batch_size_val, shuffle=True)  # shuffle the images in training set during fine-tuning
#test_dataloader   = DataLoader(test_dataset, batch_size=batch_size_val,  shuffle=False) # you don't need to shuffle test images as they are not used during training


save_path          = model_save_path_prefix + '/max_epoch_400_batch_128/alphago_policy_model_' 
loss_save_path     = model_save_path_prefix + '/max_epoch_400_batch_128/alphago_policy_model' + '_losses.mat'
train_losses       = []
test_losses        = []
train_accuracies   = []
test_accuracies    = []
start_time         = time.time()



for t in range(epochs):
    logging.info(f"Epoch {t+1}\n-------------------------------")
    avg_train_loss, train_accuracy                    = train_loop(train_dataloader, policy_network_model, loss_fn, optimizer)
    # save the losses and accuracies
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    logging.info("Epoch %d training time: %.3f minutes" %( t, (time.time()-start_time)/60))
    if t%10 == 0:     
        logging.info(f"saving the trained model at epoch{t} ...")        
        torch.save(policy_network_model.state_dict(), save_path + '_epoch_' + str(t) + '.pth')


logging.info("AlphaGoPolicyNetwork model has been trained!")
logging.info("Total training time: %.3f sec" %( (time.time()-start_time)) )
logging.info("Total training time: %.3f hrs" %( (time.time()-start_time)/3600) )
savemat(loss_save_path, {'train_losses':train_losses})

'''
# visualizing the loss curves
plt.plot(range(1,epochs+1), train_losses)
plt.plot(range(1,epochs+1), test_losses)
plt.title('AlphaGoPolicyNetwork average losses after each epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()
'''
