import torch 
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from tqdm import tqdm
import statistics 
from sklearn.preprocessing import StandardScaler
import random
#from tslearn.metrics import dtw
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fix_seed = 1111
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

plot_dir = './plots/electricity_vanilla_absolute_know_96/'  
results_dir = './results/electricity_vanilla_absolute_know_96/'

for dir_path in [plot_dir, results_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
dataset="electricity" #traffic,energy,traffic_auto,ett,electricity,nasdaq,exchange,weather 
with_knowledge=True #True,False
positional_embedding="absolute" #absolute,relative 
attention="vanilla" #vanilla,autoformer,auto_correlation 

normalization=True
test_with_attention=True

#seq_normalize_data=True 
know_different_norm=False
 
#inverse_normalize=False

sequence_length=96
horizon=96
batch_size=32
input_size=1
hidden_size=64
output_size=1
ff_hiddensize=64
mask_flag=None
attn_head=8
label_length=4
context_window=192


#traffic_auto """
"""
test_size=3509
validation_size=5263 #1754
knowledge_pred_path="./dataset/final_knowledge/final/traffic_auto/h192/t_preds.txt"
"""
#knowledge_pred_path="./dataset/final_knowledge/final/weather_preds/h192/t_preds.txt"
#knowledge_pred_path="./dataset/final_knowledge/final/nasdaq/w180_h12/t_preds.csv"
#knowledge_pred_path="./dataset/final_knowledge/final/energy/horizon_3/t_preds.csv"
#knowledge_pred_path="./dataset/final_knowledge/final/traffic_auto/h336/t_preds.txt"
#knowledge_pred_path="./dataset/final_knowledge/final/ett_preds/h96/t_preds.txt"
knowledge_pred_path="./dataset/final_knowledge/final/electricity_preds/h96/t_preds.txt"

"""
#energy 
test_size=1973
validation_size=3947
knowledge_pred_path="./dataset/final_knowledge/final/energy/horizon_6/t_preds.csv"
"""
"""
#V_228
test_size=1440
validation_size=1440*2
knowledge_pred_path="./dataset/final_knowledge/final/traffic/horizon_9/t_preds.csv"
"""

"""
#V_228
test_size=1440
validation_size=1440*2
knowledge_pred_path="./dataset/final_knowledge/final/traffic/horizon_9/t_preds.csv"
"""
#knowledge_pred_path="./dataset/final_knowledge/final/traffic/horizon_9/t_preds.csv"

if dataset=="traffic":
    df=np.asarray(pd.read_csv("./dataset/V_228.csv",header=None))
    df_know=np.asarray(pd.read_csv(knowledge_pred_path,header=None))
    test_size=1440
    validation_size=1440*2 
elif dataset=="energy":   
    df=np.asarray(pd.read_csv("./dataset/energydata_complete.txt",header=None))
    df_know=np.asarray(pd.read_csv(knowledge_pred_path,header=None))[:df.shape[0]-sequence_length]
    test_size=1973
    validation_size=3947
elif dataset=="traffic_auto":
    df = pd.read_csv("./dataset/traffic_auto/traffic.csv",
    dtype={862: 'float64'} ) 
    df=np.asarray( df.iloc[:,1:] ) 
    df_know=np.asarray(pd.read_csv(knowledge_pred_path,header=None) )
    #test_size=3509
    #validation_size=5263 #1754
    test_size=int(df.shape[0]*0.2)
    validation_size=int(df.shape[0]*0.1+test_size)
elif dataset=="ett":
    df = pd.read_csv("./dataset/final_data/ett/ett.csv") 
    df=np.asarray( df.iloc[:,1:] ) 
    df_know=np.asarray(pd.read_csv(knowledge_pred_path,header=None))
    test_size=int(df.shape[0]*0.2)
    validation_size=int(df.shape[0]*0.2 + test_size)
elif dataset=="weather":
    df = pd.read_csv("./dataset/final_data/weather/weather.csv" ) 
    df=np.asarray( df.iloc[:,1:] ) 
    df_know=np.asarray(pd.read_csv(knowledge_pred_path,header=None))
    test_size=int(df.shape[0]*0.2)
    validation_size=int(df.shape[0]*0.1+test_size)
    
    #print(f"validation_size:{validation_size} test_size:{test_size}")
elif dataset=="nasdaq":
    df=pd.read_csv("./dataset/final_data/nasdaq100_padding.csv" )
    df=np.asarray( df.iloc[:,1:] ) 
    df_know=np.asarray(pd.read_csv(knowledge_pred_path,header=None))[:df.shape[0]-sequence_length]
    test_size=int(df.shape[0]*0.1)
    validation_size=int(df.shape[0]*0.05+test_size)
elif dataset=="electricity":
    df=pd.read_csv("./dataset/final_data/electricity.csv" )
    df=np.asarray( df.iloc[:,1:] ) 
    df_know=np.asarray(pd.read_csv(knowledge_pred_path,header=None))
    test_size=int(df.shape[0]*0.2)
    validation_size=int(df.shape[0]*0.1+test_size)
elif dataset=="exchange":
    df=pd.read_csv("./dataset/final_data/exchange/exchange_rate.csv" )
    df=np.asarray( df.iloc[:,1:] )    
    df_know=np.asarray(pd.read_csv(knowledge_pred_path,header=None))              
    test_size=int(df.shape[0]*0.2)
    validation_size=int(df.shape[0]*0.1+test_size)

#print(f"type df:{type(df)} shape:{df.shape}") 
#print(f"df:{df.shape[0]}")
#print(f"df_know:{df_know.shape[0]}")


#df=df[:,:2]
#df_know=df_know[:,:2]


def make_input(data,window_size,horizon=1):
    length=data.shape[0]
    y = np.zeros([length-window_size+1-horizon,horizon])
    output=np.zeros([length-window_size+1-horizon,window_size])
    for i in range(length-window_size-horizon+1):
        output[i:i+1,:]=data[i:i+window_size]
        y[i,:]= data[i+window_size:i+window_size+horizon]
    return output.reshape(output.shape[0],window_size), y

def make_k_input(data,horizon):
    length = data.shape[0]
    output= np.zeros([length+1-horizon,horizon])
    for i in range(length-horizon+1):
        output[i:i+1,:]=data[i:i+horizon]
    return output.reshape(output.shape[0],horizon)

def nonov_make_input(data,window_size,horizon=1):
    length=data.shape[0]-window_size
    loop=length//horizon
    extra = length%horizon

    data = np.append(data,np.zeros([horizon-extra]))

    if extra ==0:
        i_val = loop
    else:
        i_val=loop+1
    
    output=np.zeros([i_val,window_size])
    y=np.zeros([i_val,horizon])
    for i in range(i_val):
        output[i:i+1,:]=data[i*horizon:(i*horizon)+window_size]
        y[i,:]= data[(i*horizon)+window_size:(i*horizon)+window_size+horizon]
        
    return output.reshape(output.shape[0],window_size), y

def nonov_make_k_input(data,horizon):
    length = data.shape[0]
    loop=length//horizon
    extra = length%horizon
    data_app = np.repeat(data[-1],(horizon-extra))
    data = np.append(data,data_app)    

    if extra ==0:
        i_val = loop
    else:
        i_val=loop+1
    output=np.zeros([i_val,horizon])
    for i in range(i_val):
        output[i:i+1,:]=data[(i*horizon):(i*horizon)+horizon]
    return output.reshape(output.shape[0],horizon)

def normalize(x):
    min_in=np.min(x,axis=1).reshape(x.shape[0],1)
    max_in=np.max(x,axis=1).reshape(x.shape[0],1) 
    denom=max_in-min_in
    a = np.where(denom == 0)[0]
    denom[a] = max_in[a] 
    a = np.where(denom == 0)[0]
    if a.size>0:
        denom[a]=1
    x=(x-min_in)/(denom)
    return x,min_in,denom

def theta_normalize(theta,min_in,denom):
      
    theta=(theta-min_in)/denom
    return theta

def metrics(pred,gt):
    l = pred.shape[1]
#     print(l)
    err_mse = np.zeros((l))
    err_mae = np.zeros((l))

    for i in range(l):
        err_mse[i] = mse(pred[:,i],gt[:,i])
        err_mae[i] = mae(pred[:,i],gt[:,i])
        
    return np.sqrt(np.mean(err_mse)),np.mean(err_mae)

def window_normalize(data_x,data_y):
    min_in = np.min(data_x,axis=1).reshape(data_x.shape[0],1)
    max_in = np.max(data_x,axis=1).reshape(data_x.shape[0],1)
    denom = (max_in-min_in)
    a = np.where(denom == 0)[0]
    denom[a] = max_in[a] 
    a = np.where(denom == 0)[0]
    if a.size >0:
        denom[a]=1
    out = (data_x-min_in)/denom
    #out=out.reshape(out.shape[0],out.shape[1],1)
    out_y=(data_y-min_in)/denom
    
    return out, out_y, denom, min_in

def window_normalize_with_expert(data_x,data_y,expert):
    
    min_in = np.min(data_x,axis=1).reshape(data_x.shape[0],1)
    max_in = np.max(data_x,axis=1).reshape(data_x.shape[0],1)
    denom = (max_in-min_in)
    
    a = np.where(denom == 0)[0]
    denom[a] = max_in[a] 
    a = np.where(denom == 0)[0]
    if a.size >0:
        denom[a]=1
    out = (data_x-min_in)/denom
    expert_normd=(expert-min_in)/denom
    
    #out=np.append(out,expert_normd,axis=1)
    #out=out.reshape(out.shape[0],out.shape[1],1)
    out_y=(data_y-min_in)/denom
    
    return out,out_y,expert_normd,denom,min_in
def window_normalize_correlation(data_x):

    min_in = np.min(data_x,axis=1).reshape(data_x.shape[0],1)
    max_in = np.max(data_x,axis=1).reshape(data_x.shape[0],1)
    denom = (max_in-min_in)
    a = np.where(denom == 0)[0]
    denom[a] = max_in[a] 
    a = np.where(denom == 0)[0]
    if a.size >0:
        denom[a]=1
    out = (data_x-min_in)/denom
    return out

def create_attention_seq(data,window_size,forecast_horizon,context_window):
  
  X_attention=np.zeros([len(data)-window_size- forecast_horizon + 1 , context_window])
  #print(X_attention.shape,len(data)-window_size- forecast_horizon + 1)
  end_index=0   
  for i in range(len(data)-window_size- forecast_horizon + 1):     
      if end_index < context_window-1:
         _x=data[0:window_size+i]  
         end_index=window_size+i
         """
         zeros_to_add=[0]*(context_window - len(_x))
         _x = pd.concat([ pd.Series(zeros_to_add),pd.Series(_x)],ignore_index=True)
         _x=_x.values  
         """         
         first_data_value = data[0]
         fill_values = [first_data_value] * (context_window - len(_x))
         _x = pd.concat([ pd.Series(fill_values),pd.Series(_x)],ignore_index=True)
         _x=_x.values
         
      else:     
         _x=data[window_size+i-context_window:window_size+i]         
      X_attention[i,:]=_x 
  return X_attention

def create_attention_test_sequences(data, window_size, horizon,context_window):
    
    length=data.shape[0]-window_size
    loop=length//horizon
    extra = length%horizon
    
    if extra ==0:
        i_val = loop
    else:
        i_val=loop+1
    
    data = np.append(data,np.zeros([horizon-extra]))    
    X_attention=np.zeros([i_val,context_window])
    
    index=0
    for i in range(i_val):
        if (index< context_window -1):
            index=(i*horizon)+window_size+horizon
            _x=data[0:(i*horizon)+window_size]
            first_data_value = data[0]
            fill_values = [first_data_value] * (context_window - len(_x))
            _x = pd.concat([ pd.Series(fill_values),pd.Series(_x)],ignore_index=True)
            _x=_x.values 
        else: 
            end_index=(i*horizon)+window_size
            _x=data[end_index-context_window:end_index]
        X_attention[i,:]=_x
        
    return X_attention

scalers=[]
know_scalers=[]
 
train_seq_x=np.zeros([df.shape[1], (len(df)-validation_size-sequence_length-horizon+1) , sequence_length ])
train_seq_y=np.zeros([df.shape[1], (len(df)-validation_size-sequence_length-horizon+1) , horizon])
valid_seq_x=np.zeros([df.shape[1], validation_size-test_size-horizon+1  , sequence_length ]) 
valid_seq_y=np.zeros([df.shape[1], validation_size-test_size-horizon+1 , horizon ])


train_attention_x=np.zeros([df.shape[1], ((len(df)-validation_size-sequence_length-horizon+1)), context_window])
valid_attention_x=np.zeros([df.shape[1],  validation_size-test_size-horizon+1, context_window])

test_seq_x=np.zeros([df.shape[1], test_size-horizon+1, sequence_length ])
test_seq_y=np.zeros([df.shape[1],  test_size-horizon+1, horizon])
test_attention_x=np.zeros([df.shape[1], test_size-horizon+1  , context_window])
test_denom=np.zeros([df.shape[1],  test_size-horizon+1, 1])
test_min=np.zeros([df.shape[1],  test_size-horizon+1, 1])
train_know_seq_y=np.zeros([df_know.shape[1], (len(df_know)-validation_size-horizon+1) , horizon])
valid_know_seq_y=np.zeros([df_know.shape[1], validation_size-test_size-horizon+1 , horizon ])
test_know_seq_y=np.zeros([df_know.shape[1],  test_size-horizon+1, horizon])

#np.set_printoptions(threshold=np.inf)

for i in range(df.shape[1]):
    current_row=df[:,i]
    train_data = current_row[:-validation_size]
    val_data = current_row[-(validation_size+sequence_length):-test_size]
    test_data = current_row[-(test_size+sequence_length):]

    know_data=df_know[:,i]
    train_know_data=know_data[:-validation_size]
    val_know_data=know_data[-validation_size:-test_size]
    test_know_data=know_data[-test_size:] 
    
    if normalization==True:
    
        scaler=StandardScaler()
        train_data=scaler.fit_transform(train_data.reshape(-1, 1))
        train_data=train_data.reshape(-1)
        scalers.append(scaler)
        
        val_data=scaler.transform(val_data.reshape(-1, 1))  
        val_data=val_data.reshape(-1) 

        test_data = scaler.transform(test_data.reshape(-1, 1))
        test_data=test_data.reshape(-1)
        
        train_sequence = make_input(train_data, sequence_length,horizon)
        val_sequence = make_input(val_data,sequence_length,horizon)
        test_sequence = make_input(test_data,sequence_length,horizon)

        
        train_know_data=scaler.transform(train_know_data.reshape(-1,1))
        train_know_data=train_know_data.reshape(-1)

        val_know_data=scaler.transform(val_know_data.reshape(-1,1))
        val_know_data=val_know_data.reshape(-1)

        test_know_data=scaler.transform(test_know_data.reshape(-1,1))
        test_know_data=test_know_data.reshape(-1)
        
        train_know_seq=make_k_input(train_know_data,horizon)
        val_know_seq=make_k_input(val_know_data,horizon)
        test_know_seq=make_k_input(test_know_data,horizon)
                
        temp_train_seq_x=train_sequence[0]
        temp_train_seq_y=train_sequence[1]
            
        temp_val_seq_x=val_sequence[0]
        temp_val_seq_y=val_sequence[1]
        
        temp_test_seq_x=test_sequence[0]
        temp_test_seq_y=test_sequence[1]
            
            
        if know_different_norm==False:
                
                temp_train_seq_x,temp_train_seq_y,temp_train_know_seq,_,_=window_normalize_with_expert(temp_train_seq_x,temp_train_seq_y,train_know_seq)
                temp_val_seq_x,temp_val_seq_y,temp_val_know_seq,_,_=window_normalize_with_expert(temp_val_seq_x,temp_val_seq_y,val_know_seq)
                temp_test_seq_x,temp_test_seq_y,temp_test_know_seq,temp_denom,temp_min=window_normalize_with_expert(temp_test_seq_x,temp_test_seq_y,test_know_seq)
                
        else:
                
                temp_train_seq_x,temp_train_seq_y,_,_=window_normalize(temp_train_seq_x,temp_train_seq_y)
                temp_val_seq_x,temp_val_seq_y,_,_=window_normalize(temp_val_seq_x,temp_val_seq_y)
                temp_test_seq_x,temp_test_seq_y,temp_denom,temp_min=window_normalize(temp_test_seq_x,temp_test_seq_y)
                
            
                temp_train_know_seq,_,_=normalize(train_know_seq)
                temp_val_know_seq,_,_=normalize(val_know_seq)
                temp_test_know_seq,_,_=normalize(test_know_seq)
            
        train_seq_x[i,:,:]=temp_train_seq_x
        train_seq_y[i,:,:]=temp_train_seq_y
        valid_seq_x[i,:,:]=temp_val_seq_x
        valid_seq_y[i,:,:]=temp_val_seq_y
        test_seq_x[i,:,:]=temp_test_seq_x
        test_seq_y[i,:,:]=temp_test_seq_y
            
        train_know_seq_y[i,:,:]= temp_train_know_seq
        valid_know_seq_y[i,:,:]= temp_val_know_seq
        test_know_seq_y[i,:,:]=temp_test_know_seq

        test_denom[i,:,:]=temp_denom
        test_min[i,:,:]=temp_min

        if attention=="auto_correlation" :         
            temp_train_attention_x=create_attention_seq(train_data.reshape(-1),sequence_length, horizon,context_window)
            train_attention_x[i,:,:]=window_normalize_correlation(temp_train_attention_x)
            temp_valid_attention_x=create_attention_seq(val_data.reshape(-1),sequence_length, horizon,context_window)
            valid_attention_x[i,:,:]=window_normalize_correlation(temp_valid_attention_x)
            temp_test_attention_x=create_attention_test_sequences(test_data.reshape(-1),sequence_length, horizon,context_window)    
            test_attention_x[i,:,:]=window_normalize_correlation(temp_test_attention_x)
            
    else:
        train_sequence = make_input(train_data, sequence_length,horizon)
        val_sequence = make_input(val_data,sequence_length,horizon)
        test_sequence = make_input(test_data,sequence_length,horizon)

        train_know_seq=make_k_input(train_know_data,horizon)
        val_know_seq=make_k_input(val_know_data,horizon)
        test_know_seq=make_k_input(test_know_data,horizon)
        
        train_seq_x[i,:,:]=train_sequence[0]
        train_seq_y[i,:,:]=train_sequence[1]
    
        valid_seq_x[i,:,:]=val_sequence[0]
        valid_seq_y[i,:,:]=val_sequence[1]
    
        test_seq_x[i,:,:]=test_sequence[0]
        test_seq_y[i,:,:]=test_sequence[1]
        
        
        train_know_seq_y[i,:,:]= train_know_seq
        valid_know_seq_y[i,:,:]= val_know_seq
        test_know_seq_y[i,:,:]=test_know_seq
        

        if attention=="auto_correlation" :         
           train_attention_x[i,:,:]=create_attention_seq(train_data.reshape(-1),sequence_length, horizon,context_window)
           valid_attention_x[i,:,:]=create_attention_seq(val_data.reshape(-1),sequence_length, horizon,context_window)
           test_attention_x[i,:,:]=create_attention_test_sequences(test_data.reshape(-1),sequence_length, horizon,context_window)
       
X_train=torch.tensor(train_seq_x,dtype=torch.float32)
y_train=torch.tensor(train_seq_y,dtype=torch.float32)
X_valid=torch.tensor(valid_seq_x,dtype=torch.float32)
y_valid=torch.tensor(valid_seq_y,dtype=torch.float32)
X_test=torch.tensor(test_seq_x,dtype=torch.float32)
y_test=torch.tensor(test_seq_y,dtype=torch.float32)

X_train=X_train.unsqueeze(-1)
X_train=X_train.view(-1,X_train.size(2),X_train.size(3))

y_train=y_train.unsqueeze(-1)
y_train=y_train.view(-1,y_train.size(2),y_train.size(3))

X_valid=X_valid.unsqueeze(-1)
X_valid=X_valid.view(-1,X_valid.size(2),X_valid.size(3))

y_valid=y_valid.unsqueeze(-1)
y_valid=y_valid.view(-1,y_valid.size(2),y_valid.size(3))

X_test=X_test.unsqueeze(-1)
y_test=y_test.unsqueeze(-1)

if normalization==True:
    Y_test_denom=torch.tensor(test_denom,dtype=torch.float32)
    Y_test_min=torch.tensor(test_min,dtype=torch.float32)
    Y_test_denom=Y_test_denom.unsqueeze(-1)
    Y_test_min=Y_test_min.unsqueeze(-1)

y_know_train_seq=torch.tensor(train_know_seq_y,dtype=torch.float32)
y_know_valid_seq=torch.tensor(valid_know_seq_y,dtype=torch.float32)
y_know_test_seq=torch.tensor(test_know_seq_y,dtype=torch.float32)
    
y_know_train_seq=y_know_train_seq.unsqueeze(-1)
Y_know_train_seq=y_know_train_seq.view(-1,y_know_train_seq.size(2),y_know_train_seq.size(3))

y_know_valid_seq=y_know_valid_seq.unsqueeze(-1)
Y_know_valid_seq=y_know_valid_seq.view(-1,y_know_train_seq.size(2),y_know_train_seq.size(3))

Y_know_test_seq=y_know_test_seq.unsqueeze(-1)

if attention=="auto_correlation":
       X_train_attention=torch.tensor(train_attention_x,dtype=torch.float32)
       X_valid_attention=torch.tensor(valid_attention_x,dtype=torch.float32)
       X_test_attention=torch.tensor(test_attention_x,dtype=torch.float32)

       X_train_attention=X_train_attention.unsqueeze(-1)
       X_train_attention=X_train_attention.view(-1,X_train_attention.size(-2),X_train_attention.size(3))

       X_valid_attention=X_valid_attention.unsqueeze(-1)
       X_valid_attention=X_valid_attention.view(-1,X_valid_attention.size(-2),X_valid_attention.size(3))

       X_test_attention=X_test_attention.unsqueeze(-1)
       
class TimeseriesDataset(Dataset):
  def __init__(self,X,y):
    self.X=X
    self.y=y
  def __len__(self):
    return len(self.X)
  def __getitem__(self,idx):
    return self.X[idx],self.y[idx]
class AttentionDataset(Dataset):
  def __init__(self,X):
    self.X=X
  def __len__(self):
    return len(self.X)
  def __getitem__(self,idx):
    return self.X[idx]
class KnowledgeDataset(Dataset):
  def __init__(self,X):
    self.X=X
  def __len__(self):
    return len(self.X)
  def __getitem__(self,idx):
    return self.X[idx]
  

train_dataset=TimeseriesDataset(X_train,y_train)
valid_dataset=TimeseriesDataset(X_valid,y_valid)
test_dataset=TimeseriesDataset(X_test,y_test)

train_loader=DataLoader(train_dataset,batch_size,drop_last=True)
valid_loader=DataLoader(valid_dataset,batch_size,drop_last=True)
test_loader=DataLoader(test_dataset,batch_size,drop_last=True)


train_know_dataset=KnowledgeDataset(Y_know_train_seq)
valid_know_dataset=KnowledgeDataset(Y_know_valid_seq)
test_know_dataset=KnowledgeDataset(Y_know_test_seq)

train_knowledge_loader=DataLoader(train_know_dataset,batch_size,drop_last=True)
valid_knowledge_loader=DataLoader(valid_know_dataset,batch_size,drop_last=True)
test_knowledge_loader=DataLoader(test_know_dataset,batch_size,drop_last=True)
        
if attention=="auto_correlation":
   train_attention_dataset=AttentionDataset(X_train_attention)
   valid_attention_dataset=AttentionDataset(X_valid_attention)
   test_attention_dataset=AttentionDataset(X_test_attention)
   
   train_attention_loader=DataLoader(train_attention_dataset,batch_size,drop_last=True)
   valid_attention_loader=DataLoader(valid_attention_dataset,batch_size,drop_last=True)
   test_attention_loader=DataLoader(test_attention_dataset,batch_size,drop_last=True)

class InputEmbedding(nn.Module):
  def __init__(self,input_size,hidden_size):
      super().__init__()
      self.input_size=input_size
      self.hidden_size=hidden_size
      self.conv1d=nn.Conv1d(in_channels=self.input_size,out_channels=self.hidden_size,padding=1,kernel_size=3,bias=False)
  def forward(self,x):
      embedded_inp=self.conv1d(x.permute(0,2,1))
      return embedded_inp.transpose(1,2)

class RelativePositionalEmbedding(nn.Module):
    def __init__(self,head_dim,max_position=512):
        super(RelativePositionalEmbedding,self).__init__()
        self.pos_embed=nn.Parameter(torch.Tensor(max_position * 2 + 1, head_dim))
        nn.init.xavier_uniform_(self.pos_embed)
        self.max_position=max_position
    def forward(self,query_len,key_len):
        query_range=torch.arange(query_len)
        key_range=torch.arange(key_len)
        relative_matrix=key_range[None,:]-query_range[:,None]
        clipped_relative_matrix = torch.clamp(relative_matrix, -self.max_position, self.max_position)
        relative_matrix=clipped_relative_matrix+self.max_position 
        return self.pos_embed[relative_matrix]
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()* -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self,x):
        return self.pe[:,:x.size(1)]  

class AttentionLayer(nn.Module):
  def __init__(self,attn_head,hidden_size):
    super().__init__()

    self.attn_head=attn_head
    self.hidden_size=hidden_size
    self.dropout = nn.Dropout(0.01)
    self.linear=nn.Linear(hidden_size,hidden_size)

    self.queries=nn.Linear(hidden_size,hidden_size)
    self.keys=nn.Linear(hidden_size,hidden_size)
    self.values=nn.Linear(hidden_size,hidden_size)

  def forward(self, queries,keys,values,attention_mask,return_attention=False):
    b,l,d=queries.shape
    b,s,d=keys.shape
    
    #Linear projection and creation of multiple heads
    queries=self.queries(queries).view(b,l,self.attn_head,-1)
    keys=self.keys(keys).view(b,s,self.attn_head,-1)
    values=self.values(values).view(b,s,self.attn_head,-1)

    b,l,h,d=queries.shape 
    b,s,h,d=values.shape

    #Calculate attention score
    attention_score=torch.einsum("blhd,bshd->bhls",queries,keys)
    
    if attention_mask == True:
       mask_shape = [b,1,l,l]
       mask=torch.triu(torch.ones(mask_shape,dtype=torch.bool),diagonal=1)
       attention_score.masked_fill_(mask.to(device),-np.inf)
    attention_score_softmax=self.dropout(torch.softmax(attention_score/sqrt(d),dim=-1))    
    final_value=torch.einsum("bhls,bshd->blhd",attention_score_softmax,values)
    weighted_attn_val=self.linear(final_value.contiguous().view(b,l,-1))

    return weighted_attn_val
  
class RelativeAttention(nn.Module):
    def __init__(self, attn_head,hidden_size):
        super(RelativeAttention, self).__init__()
        self.attn_head=attn_head
        
        self.dropout = nn.Dropout(0.01)
        self.queries=nn.Linear(hidden_size,hidden_size)
        self.keys=nn.Linear(hidden_size,hidden_size)
        self.values=nn.Linear(hidden_size,hidden_size)
        
        self.relative_positional_emd=RelativePositionalEmbedding(int(hidden_size/self.attn_head))
        self.linear=nn.Linear(hidden_size,hidden_size)
    def forward(self, query, key, value,attention_mask,return_attention=False):
        b,l,d=query.shape
        b,s,d=key.shape
        queries=self.queries(query).view(b,l,self.attn_head,-1)
        keys=self.keys(key).view(b,s,self.attn_head,-1)
        values=self.values(value).view(b,s,self.attn_head,-1)
        
        b,l,h,d=queries.shape
        b,s,h,d=values.shape
        
        a_key=self.relative_positional_emd(l,s)
        a_val=self.relative_positional_emd(l,s)
        
        qk_attention=torch.einsum("blhd,bshd->bhls",queries,keys)
        qk_relative_attention=torch.einsum(f"blhd,lsd->bhls",queries,a_key)
        
        attention_score=qk_attention+qk_relative_attention
        
        if attention_mask == True:
            
            mask_shape = [b,1,l,l]
            mask=torch.triu(torch.ones(mask_shape,dtype=torch.bool),diagonal=1)
            attention_score.masked_fill_(mask.to(device),-np.inf)
        attention_score_softmax=self.dropout(torch.softmax(attention_score/sqrt(d),dim=-1))
        
        weighted_attention=torch.einsum("bhls,bshd->blhd",attention_score_softmax,values)
        weighted_attention_rel=torch.einsum("bhls,lsd->blhd",attention_score_softmax,a_val)
        
        weighted_attention_final=weighted_attention+weighted_attention_rel
        weighted_attn_val=self.linear(weighted_attention_final.contiguous().view(b,l,-1))
        

        return weighted_attn_val

class CorrelationAttentionLayer(nn.Module):
  def __init__(self,attn_head,hidden_size): 
    super(CorrelationAttentionLayer,self).__init__()
    self.attn_head=attn_head
    self.hidden_size=hidden_size
    self.dropout=nn.Dropout(0.1)

    self.queries_emb=nn.Linear(hidden_size,hidden_size)
    self.keys_emb=nn.Linear(hidden_size,hidden_size)
    self.values_emb=nn.Linear(hidden_size,hidden_size)

  def forward(self,q,k,v,attention_mask):
    b,l,d=q.shape
    b,s,d=k.shape
    
    #Linear projection and creation of multiple heads
    queries=self.queries_emb(q).view(b,l,self.attn_head,-1)
    keys=self.keys_emb(k).view(b,s,self.attn_head,-1)
    values=self.values_emb(v).view(b,s,self.attn_head,-1)
    
    b,l,h,d=queries.shape
    b,s,h,d=values.shape
    
    if l > s:
            zeros = torch.zeros_like(queries[:, :(l - s), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
    else:
            values = values[:, :l, :, :]
            keys = keys[:, :l, :, :]

    top_k=2
    
    q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
    k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
    
    res = q_fft * torch.conj(k_fft)
    
    corr = torch.fft.irfft(res, n=l, dim=-1)
    
    mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
    top2_values, top2_indices = torch.topk(mean_value, 2, dim=1)
    #print(f"top2_indices:{top2_indices}")
    #max_value, max_index = torch.max(mean_value, dim=1)
    #print(f"max_index: {max_index}")
    
    index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
    weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)

    tmp_corr = torch.softmax(weights, dim=-1)

    values=values.permute(0, 2, 3, 1).contiguous()

    tmp_values = values
    delays_agg = torch.zeros_like(values).float()

    for i in range(top_k):
        pattern = torch.roll(tmp_values, -int(index[i]), -1)
        delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, h, d, l))
    agg_seq=delays_agg.permute(0, 3, 1, 2).view(b,l,-1)

    return agg_seq
    """
    if attention=="autoformer_context_window" and with_knowledge==True:       
        return agg_seq[:,:sequence_length+horizon+label_length+horizon,:]
        
    elif attention=="autoformer_context_window" and with_knowledge==False:
        return agg_seq[:,:label_length+horizon,:]
        
    else:
        return agg_seq
    """
"""
def autocorrelation(seq_win,con_win):
    #print(f"con_win type: {type(con_win)}")
    corr=np.correlate(con_win[i].cpu().numpy().reshape(-1),seq_win[i].cpu().numpy().reshape(-1),mode="valid")    
    return torch.tensor(corr,device=device)
"""
def get_correlation_seq(seq_win,con_win):   
    batch_size=seq_win.shape[0]
    #correlation_batch_index=np.zeros(batch_size)
    #correlation_batch_val=np.zeros(batch_size)
    correlation_seq=torch.zeros((batch_size,sequence_length+horizon,1))
    for i in range(batch_size):
        seq_win_batch=seq_win[i] 
        context_win_batch=con_win[i]
        context_win=context_win_batch[:-sequence_length,:]  
        correlation=[] 
        
        print(f"context_win shape:{context_win.shape}")
        for j in range(context_win.shape[0]-sequence_length+1):
            sequence_window=seq_win_batch 
            #context_window_data=context_win_batch[j:j+sequence_length,:]         
            #corr=autocorrelation(sequence_window,context_window_data) 
            context_window_data=context_win[j:j+sequence_length,:] 
            corr=dtw(sequence_window.cpu().numpy().reshape(-1),context_window_data.cpu().numpy().reshape(-1))
            correlation.append(corr) 
                
        min_correlation_index=torch.argmin(torch.tensor(correlation))
        
        min_correlation_index = min_correlation_index.clone().detach().to(device)
        
        correlation_seq[i]=con_win[i,min_correlation_index:min_correlation_index+sequence_length+horizon,:]
        
        l_a,_=con_win[i,min_correlation_index:min_correlation_index+sequence_length+horizon,:].shape
        if l_a < sequence_length+horizon:  
            prev_index=(sequence_length+horizon)-l_a   
            correlation_seq[i]=con_win[i,min_correlation_index-prev_index:,:]
        else:
            correlation_seq[i]=con_win[i,min_correlation_index:min_correlation_index+sequence_length+horizon,:]       
    return correlation_seq.to(device)

class Encoder(nn.Module):
  def __init__(self,attention,hidden_size,output_size):
    super(Encoder,self).__init__()
    self.attention=attention  
    self.conv1=nn.Conv1d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=1)
    self.conv2=nn.Conv1d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=1)
    self.linear=nn.Linear(hidden_size,hidden_size)
    #self.norm1=nn.LayerNorm(normalized_shape=(sequence_length,hidden_size))
    self.norm1=nn.LayerNorm(hidden_size)
    self.activation=F.relu
    self.dropout=nn.Dropout(0.01)
    
    self.queries=nn.Linear(hidden_size,hidden_size)
    self.keys=nn.Linear(hidden_size,hidden_size)
    self.values=nn.Linear(hidden_size,hidden_size)
    """
    #For autoformer with context_window, the attention returned will be downsampled to match the size
    #of input sequence so that it can be added as a residual connection  :
    if with_knowledge==True:
        self.downsample_layer=nn.Linear(context_window,sequence_length+horizon)
    else:
        self.downsample_layer=nn.Linear(context_window,sequence_length)
    """
  def forward(self,x,x_attn=None):
        
    if test_with_attention==False:
        #Skipping attention and passed through FFN only
        out=self.conv1(x.permute(0,2,1))
        out=self.activation(out)
        out=self.dropout(self.conv2(out).transpose(-1, 1))
        norm_out=self.norm1(out)
        return norm_out 
    else:
        if attention=="auto_correlation": 
            #pass the most correlated sequence_window + the horizon following it
            #to get the attention 
            attention_x=self.attention(x_attn,x_attn,x_attn,attention_mask=False)
            #skipped residual connectoin here, otherwise to match the sizes of attention with context
            # and original input (window_size), we had to downsample attention(previous_datapoints+window_size) output .
            new_x=attention_x 
        else:
            attention_x=self.attention(x,x,x,attention_mask=False) 
            new_x = x + attention_x      
        res_x=x=self.norm1(new_x)
        ##Feed forward NN: 
        out=self.conv1(res_x.permute(0,2,1))
        out=self.dropout(self.activation(out)) 
        out=self.dropout(self.conv2(out).transpose(-1, 1))

        ##Add and normalize:
        new_out=out+res_x
        norm_out=self.norm1(new_out)
        
        return norm_out

class Decoder(nn.Module):
  def __init__(self,attention,hidden_size,output_size):
    super(Decoder,self).__init__()
    self.attention=attention
    self.conv1=nn.Conv1d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=1)
    self.conv2=nn.Conv1d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=1)

    self.linear=nn.Linear(hidden_size,output_size)
    self.dropout=nn.Dropout(0.01)
    self.norm1=nn.LayerNorm(hidden_size)
    self.norm2=nn.LayerNorm(hidden_size)
    self.norm3=nn.LayerNorm(hidden_size)
    self.activation=F.relu

  def forward(self,dec_inp,enc_out,know_pred):
    
    if test_with_attention==False:
        
        #FFN
        out=self.conv1(enc_out.permute(0,2,1))
        out=self.activation(out)
        out=self.dropout(self.conv2(out).transpose(-1, 1))
        out=self.norm3(out)
        
        #Linear projection
        pred=self.linear(out)
        return pred
    
    else:  
        
        self_attn=self.attention(dec_inp,dec_inp,dec_inp,attention_mask=True)
        #add residual connection and normalize
        
        residual_add=dec_inp + self_attn
        new_dec_x=self.norm1(residual_add)
    
        # encoder-decoder attention. Pass encoder output as key and value and queries as output of self-attention of decoder
        enc_dec_atten=self.attention(new_dec_x,enc_out,enc_out,attention_mask=False)
        
        ## add and normalize
        new_x=enc_dec_atten+self_attn
        norm_x=self.norm2(new_x)
    
        #FFN
        out=self.conv1(norm_x.permute(0,2,1))
        out=self.dropout(self.activation(out))
        out=self.dropout(self.conv2(out).transpose(-1, 1))

        #add and normalize
        new_x=out+norm_x
        out=self.norm3(new_x)
        
        #if with_knowledge==True:
        #   out=out+dec_inp
        #Linear projection
        pred=self.linear(out)
        pred=pred[:,-horizon:,:]+know_pred
        return pred

class TransformerModel(nn.Module):
  def __init__(self,input_size,hidden_size,output_size,attn_head):
      super(TransformerModel,self).__init__()
      
      self.enc_embedding=InputEmbedding(input_size,hidden_size)
      self.dec_embedding=InputEmbedding(input_size,hidden_size)
        
      if positional_embedding=="absolute":  
            self.enc_positional_embedding=PositionalEmbedding(hidden_size)
            self.dec_positional_embedding=PositionalEmbedding(hidden_size)
            if attention=="vanilla" or attention=="auto_correlation":
                    self.encoder=Encoder( AttentionLayer(attn_head,hidden_size),hidden_size,output_size)
                    self.decoder=Decoder( AttentionLayer(attn_head,hidden_size),hidden_size,output_size)
            elif attention=="autoformer": 
                    self.encoder=Encoder( CorrelationAttentionLayer(attn_head,hidden_size),hidden_size,output_size)  
                    self.decoder=Decoder( CorrelationAttentionLayer(attn_head,hidden_size),hidden_size,output_size)
            
      else:    
            #Uses Relative positional embedding, used during vanilla self attention
            self.encoder=Encoder( RelativeAttention(attn_head,hidden_size),hidden_size,output_size)
            self.decoder=Decoder( RelativeAttention(attn_head,hidden_size),hidden_size,output_size)
            
            
            #self.encoders = nn.ModuleList([Encoder(AttentionLayer(self.attn_head, hidden_size), hidden_size, output_size, ff_hiddensize, sequence_length)
                                        #for _ in range(2)])
            #self.decoders = nn.ModuleList([Decoder( AttentionLayer(self.attn_head,hidden_size),self.hidden_size,self.output_size,self.ff_hidden_size,sequence_length )
                                        #for _ in range(2)])
                
      self.linear=nn.Linear(hidden_size,output_size)
      self.dropout=nn.Dropout(0.01) 
      
  def forward(self,x,y,x_attn=None,know_pred=None):
       
        if know_pred is not None:  
            #1.For encoder, Integrate knowledge predictions to the sequence_window(x) to get the context vector at the encoder that has- 
            #-informataion of future using the knowledge predictions
            #2.For decoder, just the previous four data points from the forecast horizon x[:,-4:,:]
            #are included. 
            
            encoder_input=torch.cat((x,know_pred),dim=1)
            decoder_input=torch.cat((x[:,-4:,:],know_pred),dim=1)
        else:
            encoder_input=x
            decoder_input=torch.cat((x[:,-4:,:],torch.zeros_like(y[:,-horizon:,:])),dim=1)
      
        #ENCODER
        
        #decide here which positional embedding
        if positional_embedding=="absolute":
            inp_embed=self.enc_embedding(encoder_input)
            pos_embed=self.enc_positional_embedding(encoder_input)
            enc_out=inp_embed + pos_embed
            
            inp_embed=self.dec_embedding(decoder_input)
            pos_embed=self.dec_positional_embedding(decoder_input)
            dec_out= inp_embed + pos_embed
            """
            if attention=="auto_correlation":
                inp_embed=self.enc_embedding(x_attn)
                pos_embed=self.enc_positional_embedding(x_attn)
                enc_out_attn_context=inp_embed + pos_embed 
            """
        else:  
            # In relative positional embedding, the positional embedding are learnt on the fly by the model during attention. Used Shaw et al., method,
            # hence, there is only input embedding,positional embedding part in attention(RelativeAttention).
            enc_out=self.enc_embedding(encoder_input)
            dec_out=self.dec_embedding(decoder_input)
        
        # After input embedding check which attention must be used.
        if attention=="auto_correlation":
            #get the most correlated sequence+horizon of "X" in "X_attn(Context_window)"
            corr_seq=get_correlation_seq(x,x_attn) 
            
            attn_inp_embed=self.enc_embedding(corr_seq)
            attn_pos_embed=self.enc_positional_embedding(corr_seq)
            
            attn_enc_out=attn_inp_embed+attn_pos_embed  
            enc_out=self.encoder(enc_out,attn_enc_out)        
        else:
          # if attention is vanilla attention/autoformer attention:
            enc_out=self.encoder(enc_out)           
        #decoder      
        out=self.decoder(dec_out,enc_out,know_pred)        
        return out 
model=TransformerModel(input_size,hidden_size,output_size,attn_head)


model=model.to(device)
loss_fun=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)
epochs=5

for epoch in range(epochs):
        train_loss=[]
        valid_total_loss=[]
        model.train()
        
        if with_knowledge==False and (attention=="auto_correlation"):
              
              for (batch_idx, (X,y)),(batch_idx_3,(x_attention)) in zip(enumerate(train_loader), enumerate(train_attention_loader)):
                    
                    #1.Pass the window,forecast horizon,knowledge prediction and a context window of say 120 time points-
                    #-that are present upto the end of current window.
                    #2.Get the most correlated sequence index and the data points of window_size after the correlated index (np.correlate (context_window,sequence_window)) 
                    #3.Pass this correlated sequence as keys and values to the attention and sequence_window as query in attention mechanism
                    
                    pred=model(X.to(device),y.to(device),x_attention.to(device))
                    pred=pred[:,-horizon:,:]
                    optimizer.zero_grad()
                    loss=loss_fun(pred,y.to(device))
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())
              train_loss = np.average(train_loss)      
        
              model.eval()
              with torch.no_grad():
               for (batch_idx, (X,y)),(batch_idx_3,(x_attention)) in zip(enumerate(valid_loader), enumerate(valid_attention_loader)):
                    pred=model(X.to(device),y.to(device),x_attention.to(device))
                    pred=pred[:,-horizon:,:]
            
                    valid_loss=loss_fun(pred,y.to(device)) 
                    valid_total_loss.append(valid_loss.item())
               valid_total_loss=np.average(valid_total_loss)
              print(f"epoch:{epoch} train_loss:{train_loss} valid_loss:{valid_total_loss}") 

        elif with_knowledge == True and (attention =="vanilla" or attention=="autoformer"):  
              attn=None
              # Simple vanilla transformer with knowledge prediction integrated
              for (batch_idx, (X,y)),(batch_idx_2,(know_pred)) in zip(enumerate(train_loader), enumerate(train_knowledge_loader)): 
                    pred=model(X.to(device),y.to(device),attn,know_pred.to(device))
                    pred=pred[:,-horizon:,:]
                    optimizer.zero_grad()
                    loss=loss_fun(pred,y.to(device))
                    loss.backward()
                    optimizer.step()
                    
                    train_loss.append(loss.item())
            
              train_loss = np.average(train_loss)
        
              model.eval()
        
              with torch.no_grad():
               
               for (batch_idx, (X,y)),(batch_idx_2,(know_pred)) in zip(enumerate(valid_loader), enumerate(valid_knowledge_loader)):
                   pred=model(X.to(device),y.to(device),attn,know_pred.to(device))
                   pred=pred[:,-horizon:,:]
    
                   valid_loss=loss_fun(pred,y.to(device)) 
                   valid_total_loss.append(valid_loss.item())
            
               valid_total_loss=np.average(valid_total_loss)          
              print(f"epoch:{epoch} train_loss:{train_loss} valid_loss:{valid_total_loss}") 

        elif with_knowledge == False and (attention =="vanilla" or attention=="autoformer"): 

            for (batch_idx, (X,y)) in enumerate(train_loader): 
                    pred=model(X.to(device),y.to(device))
                    pred=pred[:,-horizon:,:]
                    optimizer.zero_grad()
                    loss=loss_fun(pred,y.to(device))
                    loss.backward()
                    optimizer.step()
                     
                    train_loss.append(loss.item())
            
            train_loss = np.average(train_loss)
        
            model.eval()
        
            with torch.no_grad():
          
                for (batch_idx, (X,y)) in enumerate(valid_loader):
                   pred=model(X.to(device),y.to(device))
                   pred=pred[:,-horizon:,:]
                   
                   valid_loss=loss_fun(pred,y.to(device)) 
                   valid_total_loss.append(valid_loss.item())
            
                valid_total_loss=np.average(valid_total_loss)      
            print(f"epoch:{epoch} train_loss:{train_loss} valid_loss:{valid_total_loss}")  


output=[]
loss=[]
results=[]


model.eval()

for i in range(X_test.size(0)):
  prediction=[]
  with torch.no_grad(): 
    if with_knowledge==False and (attention=="auto_correlation"):
            
            current_X_test=X_test[i,:,:,:]  
            current_y_test=y_test[i,:,:,:]
            current_X_attention=X_test_attention[i,:,:,:]

            """           
            pred=model(current_X_test.to(device),current_y_test.to(device),current_X_attention.to(device))
            pred=pred[:,-horizon:,:]
            """
            X_test_dataset = TimeseriesDataset(current_X_test, current_y_test)
            X_test_dataloader = DataLoader(X_test_dataset, batch_size=batch_size, drop_last=False)
            
            X_attention_dataset=AttentionDataset(current_X_attention)
            X_attention_dataloader=DataLoader(X_attention_dataset,batch_size=batch_size,drop_last=False)

            for (batch_idx, (X,y)),(batch_idx_2,(X_attn)) in zip(enumerate(X_test_dataloader), enumerate(X_attention_dataloader)): 
               
               preds=model(X.to(device),y.to(device),X_attn.to(device))
               preds=preds[:,-horizon:,:]
               prediction.append(preds)
               #torch.cuda.empty_cache()
               
            pred = torch.cat(prediction, dim=0)
            

            if normalization==True:
               current_Y_test_denom=Y_test_denom[i,:,:,:]
               current_Y_test_min=Y_test_min[i,:,:,:]
               pred=pred*current_Y_test_denom.to(device)+current_Y_test_min.to(device)
               current_y_test=current_y_test.to(device)*current_Y_test_denom.to(device)+current_Y_test_min.to(device)
               pred=pred.reshape(-1,1).detach().cpu().numpy()
               current_y_test=current_y_test.reshape(-1,1).detach().cpu().numpy() 
               #pred=scalers[i].inverse_transform(pred.reshape(-1,1).detach().cpu().numpy())
               #current_y_test=scalers[i].inverse_transform(current_y_test.reshape(-1,1).detach().cpu().numpy())
            else:
               pred=pred.reshape(-1,1).detach().cpu().numpy()
               current_y_test=current_y_test.reshape(-1,1).detach().cpu().numpy()
            
            df = pd.DataFrame({'Predictions': pred.flatten(),'Ground Truth': current_y_test.flatten()})
            file_name = os.path.join(results_dir, f"series{i}.csv")
            df.to_csv(file_name, index=False)
             
            loss.append(loss_fun(torch.tensor(pred),torch.tensor(current_y_test)))               
            
            
            plt.figure(figsize=(10, 6))
            plt.plot(current_y_test, label='Ground Truth')
            plt.plot(pred, label='Predicted')
            plt.title(f'Time Series {i+1}: Ground Truth vs Predicted Values')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            #plt.show()
            plt.savefig(os.path.join(plot_dir, f'Time_Series_{i+1}_plot.png'))
            plt.close()
            
    elif with_knowledge==True and (attention=="autoformer" or attention=="vanilla"):
            attn=None
            current_X_test=X_test[i,:,:,:]
            current_know_pred=Y_know_test_seq[i,:,:,:]
            current_y_test=y_test[i,:,:,:]
            
            #pred=model(current_X_test.to(device),current_y_test.to(device),current_know_pred.to(device))
            #pred=pred[:,-horizon:,:]

            
            X_test_dataset = TimeseriesDataset(current_X_test, current_y_test)
            X_test_dataloader = DataLoader(X_test_dataset, batch_size=batch_size, drop_last=False)
            
            X_test_knowledge_dataset = KnowledgeDataset(current_know_pred)
            X_test_knowledge_dataloader=DataLoader(X_test_knowledge_dataset,batch_size=batch_size,drop_last=False)
            
            
            for (batch_idx, (X,y)),(batch_idx_2,(know_pred)) in zip(enumerate(X_test_dataloader), enumerate(X_test_knowledge_dataloader)): 
               
               preds=model(X.to(device),y.to(device),attn,know_pred.to(device))
               preds=preds[:,-horizon:,:]
               prediction.append(preds)
               #torch.cuda.empty_cache()
               
            pred = torch.cat(prediction, dim=0) 
            #print(f"pred shape:{pred.shape}") 
            if normalization==True:
                  current_Y_test_denom=Y_test_denom[i,:,:,:]
                  current_Y_test_min=Y_test_min[i,:,:,:]
                  pred=pred*current_Y_test_denom.to(device)+current_Y_test_min.to(device)
                  current_y_test=current_y_test.to(device)*current_Y_test_denom.to(device)+current_Y_test_min.to(device)
                  pred=pred.reshape(-1,1).detach().cpu().numpy()
                  current_y_test=current_y_test.reshape(-1,1).detach().cpu().numpy() 
                  #pred=scalers[i].inverse_transform(pred.reshape(-1,1).detach().cpu().numpy())
                  #current_y_test=scalers[i].inverse_transform(current_y_test.reshape(-1,1).detach().cpu().numpy())
            else:
                 pred=pred.reshape(-1,1).detach().cpu().numpy()
                 current_y_test=current_y_test.reshape(-1,1).detach().cpu().numpy()
               
            df = pd.DataFrame({'Predictions': pred.flatten(),'Ground Truth': current_y_test.flatten()})
            file_name = os.path.join(results_dir, f"series{i}.csv")
            df.to_csv(file_name, index=False)

            loss.append(loss_fun(torch.tensor(pred),torch.tensor(current_y_test)))
                       
            plt.figure(figsize=(10, 6))
            plt.plot(current_y_test, label='Ground Truth')
            plt.plot(pred, label='Predicted')
            plt.title(f'Time Series {i+1}: Ground Truth vs Predicted Values')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            #plt.show()
            plt.savefig(os.path.join(plot_dir, f'Time_Series_{i+1}_plot.png'))
            plt.close()
            torch.cuda.empty_cache() 
    #Just vanilla        
    elif with_knowledge==False:
            
            current_X_test=X_test[i,:,:,:]
            current_y_test=y_test[i,:,:,:]
           
            X_test_dataset = TimeseriesDataset(current_X_test, current_y_test)
            X_test_dataloader = DataLoader(X_test_dataset, batch_size=batch_size, drop_last=False)
            
            for (batch_idx, (X,y)) in enumerate(X_test_dataloader): 
               
               preds=model(X.to(device),y.to(device))
               preds=preds[:,-horizon:,:]
               prediction.append(preds)
            pred = torch.cat(prediction, dim=0)
            """
            pred=model(current_X_test.to(device),current_y_test.to(device))
            pred=pred[:,-horizon:,:]
            """ 
            if normalization==True:
               current_Y_test_denom=Y_test_denom[i,:,:,:]
               current_Y_test_min=Y_test_min[i,:,:,:] 
               pred=pred*current_Y_test_denom.to(device)+current_Y_test_min.to(device)
               current_y_test=current_y_test.to(device)*current_Y_test_denom.to(device)+current_Y_test_min.to(device)
               pred=pred.reshape(-1,1).detach().cpu().numpy()
               current_y_test=current_y_test.reshape(-1,1).detach().cpu().numpy() 
               #pred=scalers[i].inverse_transform(pred.reshape(-1,1).detach().cpu().numpy())
               #current_y_test=scalers[i].inverse_transform(current_y_test.reshape(-1,1).detach().cpu().numpy())
            else:
               pred=pred.reshape(-1,1).detach().cpu().numpy()
               current_y_test=current_y_test.reshape(-1,1).detach().cpu().numpy()
            #np.set_printoptions(threshold=np.inf) 
            #print(f"pred:{pred},y_test:{current_y_test}")   
             
            df = pd.DataFrame({'Predictions': pred.flatten(),'Ground Truth': current_y_test.flatten()})
            file_name = os.path.join(results_dir, f"series{i}.csv")
            df.to_csv(file_name, index=False)
            
            loss.append(loss_fun(torch.tensor(pred),torch.tensor(current_y_test)))
            

            plt.figure(figsize=(10, 6))
            plt.plot(current_y_test, label='Ground Truth')
            plt.plot(pred, label='Predicted')
            plt.title(f'Time Series {i+1}: Ground Truth vs Predicted Values')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            #plt.show()
            plt.savefig(os.path.join(plot_dir, f'Time_Series_{i+1}_plot.png'))
            plt.close()
            
loss_val=torch.stack(loss,dim=0)
mean_loss=torch.mean(loss_val)

print(f"mean loss:{mean_loss}, loss_val:{loss_val}")

file_name = os.path.join(results_dir, 'series0.csv')
df = pd.read_csv(file_name)

ground_truth = df['Ground Truth'][:200]
predictions = df['Predictions'][:200]
plt.figure(figsize=(10, 6)) 
plt.plot(ground_truth, label='Ground Truth')
plt.plot(predictions, label='Predictions')
plt.xlabel('Time points')
plt.ylabel('Values')
plt.title('Ground Truth vs Predictions')
plt.legend()
plt.savefig(os.path.join(plot_dir, f'series_0.png'))
plt.show()