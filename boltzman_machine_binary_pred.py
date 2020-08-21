import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn 
import torch.nn.parallel 
import torch.optim as optim 
import torch.utils.data 
from torch.autograd import Variable 

movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') 
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') 
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') 

training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

nbr_u = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nbr_m = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

def convert(data):
    new_data = []
    for user_id in range(1, nbr_u + 1):
        movies_id = data[:, 1][data[:, 0] == user_id]
        ratings_id = data[:, 2][data[:, 0] == user_id]
        ratings = np.zeros(nbr_m)
        ratings[movies_id - 1] = ratings_id
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)
 
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# converting ratings to binary value (1, 0, -1)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)

    def sample_h(self, x):
        wx = torch.mm(x,  self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y,  self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
    
nv = len(training_set[0]) 
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

#train
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    counter = 0.
    for user_id in range(0, nbr_u - batch_size, batch_size):
        vk = training_set[user_id : user_id + batch_size]
        v0 = training_set[user_id : user_id + batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0] 
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        counter += 1.
        
    print('epoch: '+str(epoch)+' train_loss: '+str(train_loss/counter))       
      
#test
test_loss = 0
counter = 0.
for user_id in range(nbr_u):
    v = training_set[user_id : user_id+1]
    vt = test_set[user_id : user_id+1]
    ph0,_ = rbm.sample_h(v0)
    if len(vt[[vt>=0]]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        counter += 1.
        
print('test_loss: '+str(test_loss/counter))           
            
            
            
            
            
            

