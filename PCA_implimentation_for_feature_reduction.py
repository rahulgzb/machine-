import numpy as np
import torch as T
import pprint
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
device = T.device("cpu")  # apply to Tensor or Module



#data preprocessing
df= pd.read_csv('diabetes.csv')
x=df.drop('Outcome',axis =1)
y=df['Outcome']
print("Before implimenting PCA train data shape:")
print(x.shape)
pca = PCA(n_components =2)
X_pca = pca.fit_transform(x)
print("After implimenting PCA train data shape:")
print(X_pca.shape)
x=X_pca
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.3)


class trainDataset(T.utils.data.Dataset):

  def __init__(self, src_file, num_rows=None):
    all_data = np.loadtxt(src_file, max_rows=num_rows,
      usecols=range(1,9), delimiter=",", skiprows=1,
      dtype=np.float32)  # strip IDs off
      

    self.x_data = T.tensor(X_train[:,0:2],
      dtype=T.float32).to(device)
      

    Y_train= y_train.values 
      

    self.y_data = T.tensor(Y_train[:,],
      dtype=T.float32).to(device)
      
    self.y_data = self.y_data.reshape(-1,1)
  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    if T.is_tensor(idx):
      idx = idx.tolist()
    preds = self.x_data[idx,:]  # idx rows, all 
    lbl = self.y_data[idx,:]    # idx rows, the
    sample = { 'predictors' : preds, 'target' : lbl }

    return sample

class test_data_set(T.utils.data.Dataset):

  def __init__(self, src_file, num_rows=None):
    all_data = np.loadtxt(src_file, max_rows=num_rows,
      usecols=range(1,9), delimiter=",", skiprows=1,
      dtype=np.float32)  # strip IDs off
    

    self.x_data = T.tensor(X_test[:,0:2],
      dtype=T.float32).to(device)
    Y_test = y_test.values  
    self.y_data = T.tensor(Y_test[:,],
      dtype=T.float32).to(device)

    # n_vals = len(self.y_data)
    # self.y_data = self.y_data.reshape(n_vals,1)
    self.y_data = self.y_data.reshape(-1,1)
  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    if T.is_tensor(idx):
      idx = idx.tolist()
    preds = self.x_data[idx,:]  # idx rows, all 4 cols
    lbl = self.y_data[idx,:]    # idx rows, the 1 col
    sample = { 'predictors' : preds, 'target' : lbl }
    # sample = dict()   # or sample = {}
    # sample["predictors"] = preds
    # sample["target"] = lbl

    return sample
# ---------------------------------------------------------



#HELPER FUNCTIONS

def accuracy(model, ds):
  # ds is a iterable Dataset of Tensors
  n_correct = 0; n_wrong = 0

  # alt: create DataLoader and then enumerate it
  for i in range(len(ds)):
    inpts = ds[i]['predictors']
    target = ds[i]['target']    # float32  [0.0] or [1.0]
    with T.no_grad():
      oupt = model(inpts)

    # avoid 'target == 1.0'
    if target < 0.5 and oupt < 0.5:  # .item() not needed
      n_correct += 1
    elif target >= 0.5 and oupt >= 0.5:
      n_correct += 1
    else:
      n_wrong += 1

  return (n_correct * 1.0) / (n_correct + n_wrong)

# ---------------------------------------------------------

def acc_coarse(model, ds):
  inpts = ds[:]['predictors']  # all rows
  targets = ds[:]['target']    # all target 0s and 1s
  with T.no_grad():
    oupts = model(inpts)         # all computed ouputs
  pred_y = oupts >= 0.5        # tensor of 0s and 1s
  num_correct = T.sum(targets==pred_y)
  acc = (num_correct.item() * 1.0 / len(ds))  # scalar
  return acc

# ----------------------------------------------------------

def my_bce(model, batch):
  
  sum = 0.0
  inpts = batch['predictors']
  targets = batch['target']
  with T.no_grad():
    oupts = model(inpts)
  for i in range(len(inpts)):
    oupt = oupts[i]
    # should prevent log(0) which is -infinity
    if targets[i] >= 0.5:  # avoiding == 1.0
      sum += T.log(oupt)
    else:
      sum += T.log(1 - oupt)

  return -sum / len(inpts)

# ----------------------------------------------------------
# model with 1 hidden layer and 6 Nodes 

class Net0(T.nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.oupt = T.nn.Linear(2, 1)

        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = T.sigmoid(self.oupt(x))
        return z


class Net1(T.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.hid1 = T.nn.Linear(2, 6)  # 7-6-1
        self.oupt = T.nn.Linear(6, 1)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = T.tanh(self.hid1(x))  # or T.nn.Tanh()
        z = T.sigmoid(self.oupt(z))
        return z


class Net2(T.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.hid1 = T.nn.Linear(2, 2)
        self.oupt = T.nn.Linear(2, 1)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = T.tanh(self.hid1(x))  # or T.nn.Tanh()
        z = T.sigmoid(self.oupt(z))
        return z


class Net3(T.nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.hid1 = T.nn.Linear(2, 3)
        self.hid2 = T.nn.Linear(3, 2)
        self.oupt = T.nn.Linear(2, 1)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = T.tanh(self.hid1(x))  # or T.nn.Tanh()
        z = T.tanh(self.hid2(z))
        z = T.sigmoid(self.oupt(z))  #
        return z


class Net4(T.nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.hid1 = T.nn.Linear(2, 2)  #
        self.hid2 = T.nn.Linear(2, 3)
        self.oupt = T.nn.Linear(3, 1)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = T.tanh(self.hid1(x))  # or T.nn.Tanh()
        z = T.tanh(self.hid2(z))
        z = T.sigmoid(self.oupt(z))  # 
        return z

# ----------------------------------------------------------



def main():
  # 0. get started
  
  T.manual_seed(1)
  np.random.seed(1)

  # 1. create Dataset and DataLoader objects
  print("------MLP with reduced Feature space--------")

 

  train_file = "diabetes.csv"
  test_file = "diabetes.csv"

  train_ds = trainDataset(train_file)  # all rows
  

# get first sample and unpack
  
  test_ds = test_data_set(test_file)

  bat_size = 10
  train_ldr = T.utils.data.DataLoader(train_ds,
    batch_size=bat_size, shuffle=True)
 
 # implimentation of all the models  
  print("\nModel-1")
  print("0 Hidden layer")
  net = Net1().to(device)

  # 3. train network
  net = net.train()  # set training mode
  lrn_rate = 0.1

  loss_obj = T.nn.MSELoss()  # Mse loss function 
  optimizer = T.optim.SGD(net.parameters(),
    lr=lrn_rate)
  max_epochs = 100
  ep_log_interval = 10
  print("Loss function: " + str(loss_obj))
  print("Optimizer: SGD")
  print("Best Learn rate: 0.1")
  print("Batch size: 10")
  print("Max epochs: " + str(max_epochs))

  print("\nStarting training")
  for epoch in range(0, max_epochs):
    epoch_loss = 0.0            # for one full epoch
    epoch_loss_custom = 0.0
    num_lines_read = 0

    for (batch_idx, batch) in enumerate(train_ldr):
      X = batch['predictors']  # inputs
      Y = batch['target']      #   targets
      oupt = net(X)            #   computeds 

      loss_val = loss_obj(oupt, Y)   # a tensor
      epoch_loss += loss_val.item()  # accumulate
      

      optimizer.zero_grad() # reset all gradients
      loss_val.backward()   # compute all gradients
      optimizer.step()      # update all weights

    if epoch % ep_log_interval == 0:
      print("epoch = %4d   loss = %0.4f" % \
        (epoch, epoch_loss))
      # print("custom loss = %0.4f" % epoch_loss_custom)
      # print("")
  print("Done ")

# ----------------------------------------------------------

  # 4. evaluate model
  net = net.eval()
  acc_train = accuracy(net, train_ds)
  print("\nAccuracy on train data = %0.2f%%" % \
    (acc_train * 100))
  acc_test = accuracy(net, test_ds)
  print("Accuracy on test data = %0.2f%%" % \
    (acc_test * 100))
#model 2 for best accuracy
  print("\nModel-2")
  print("1 Hidden layer with 6 Nodes")
  net = Net1().to(device)

  # 3. train network
  net = net.train()  # set training mode
  lrn_rate = 0.1

  loss_obj = T.nn.MSELoss()  # Mse loss function 
  optimizer = T.optim.SGD(net.parameters(),
    lr=lrn_rate)
  max_epochs = 100
  ep_log_interval = 10
  print("Loss function: " + str(loss_obj))
  print("Optimizer: SGD")
  print("Best Learn rate: 0.1")
  print("Batch size: 10")
  print("Max epochs: " + str(max_epochs))

  print("\nStarting training")
  for epoch in range(0, max_epochs):
    epoch_loss = 0.0            # for one full epoch
    epoch_loss_custom = 0.0
    num_lines_read = 0

    for (batch_idx, batch) in enumerate(train_ldr):
      X = batch['predictors']  # inputs
      Y = batch['target']      #   targets
      oupt = net(X)            #   computeds 

      loss_val = loss_obj(oupt, Y)   # a tensor
      epoch_loss += loss_val.item()  # accumulate

      optimizer.zero_grad() # reset all gradients
      loss_val.backward()   # compute all gradients
      optimizer.step()      # update all weights

    if epoch % ep_log_interval == 0:
      print("epoch = %4d   loss = %0.4f" % \
        (epoch, epoch_loss))
      
  print("Done ")

# ----------------------------------------------------------

  # 4. evaluate model
  net = net.eval()
  acc_train = accuracy(net, train_ds)
  print("\nAccuracy on train data = %0.2f%%" % \
    (acc_train * 100))
  acc_test = accuracy(net, test_ds)
  print("Accuracy on test data = %0.2f%%" % \
    (acc_test * 100))


  print("\nModel-3")
  print("1 Hidden layer with 2 Nodes")
  net = Net2().to(device)

  # 3. train network
  net = net.train()  # set training mode
  lrn_rate = 0.1

  loss_obj = T.nn.MSELoss()  # Mse loss function 
  optimizer = T.optim.SGD(net.parameters(),
    lr=lrn_rate)
  max_epochs = 100
  ep_log_interval = 10
  print("Loss function: " + str(loss_obj))
  print("Optimizer: SGD")
  print("Best Learn rate: 0.1")
  print("Batch size: 10")
  print("Max epochs: " + str(max_epochs))

  print("\nStarting training")
  for epoch in range(0, max_epochs):
    epoch_loss = 0.0            # for one full epoch
    epoch_loss_custom = 0.0
    num_lines_read = 0

    for (batch_idx, batch) in enumerate(train_ldr):
      X = batch['predictors']  # inputs
      Y = batch['target']      #   targets
      oupt = net(X)            #   computeds 

      loss_val = loss_obj(oupt, Y)   # a tensor
      epoch_loss += loss_val.item()  # accumulate
     

      optimizer.zero_grad() # reset all gradients
      loss_val.backward()   # compute all gradients
      optimizer.step()      # update all weights

    if epoch % ep_log_interval == 0:
      print("epoch = %4d   loss = %0.4f" % \
        (epoch, epoch_loss))
      
  print("Done ")

# ----------------------------------------------------------

  # 4. evaluate model
  net = net.eval()
  acc_train = accuracy(net, train_ds)
  print("\nAccuracy on train data = %0.2f%%" % \
    (acc_train * 100))
  acc_test = accuracy(net, test_ds)
  print("Accuracy on test data = %0.2f%%" % \
    (acc_test * 100))

  "model 4 code"

  print("\nModel-4")
  print("2 Hidden layer with 3,2 Nodes")
  net = Net3().to(device)

  # 3. train network
  net = net.train()  # set training mode
  lrn_rate = 0.1

  loss_obj = T.nn.MSELoss()  # Mse loss function 
  optimizer = T.optim.SGD(net.parameters(),
    lr=lrn_rate)
  max_epochs = 100
  ep_log_interval = 10
  print("Loss function: " + str(loss_obj))
  print("Optimizer: SGD")
  print("Learn rate: 0.1")
  print("Batch size: 10")
  print("Max epochs: " + str(max_epochs))

  print("\nStarting training")
  for epoch in range(0, max_epochs):
    epoch_loss = 0.0            # for one full epoch
    epoch_loss_custom = 0.0
    num_lines_read = 0

    for (batch_idx, batch) in enumerate(train_ldr):
      X = batch['predictors']  # inputs
      Y = batch['target']      #   targets
      oupt = net(X)            #   computeds 

      loss_val = loss_obj(oupt, Y)   # a tensor
      epoch_loss += loss_val.item()  # accumulate
      # epoch_loss += loss_val  # is OK
      # epoch_loss_custom += my_bce(net, batch)

      optimizer.zero_grad() # reset all gradients
      loss_val.backward()   # compute all gradients
      optimizer.step()      # update all weights

    if epoch % ep_log_interval == 0:
      print("epoch = %4d   loss = %0.4f" % \
        (epoch, epoch_loss))
      # print("custom loss = %0.4f" % epoch_loss_custom)
      # print("")
  print("Done ")

# ----------------------------------------------------------

  # 4. evaluate model
  net = net.eval()
  acc_train = accuracy(net, train_ds)
  print("\nAccuracy on train data = %0.2f%%" % \
    (acc_train * 100))
  acc_test = accuracy(net, test_ds)
  print("Accuracy on test data = %0.2f%%" % \
    (acc_test * 100))
  "model 5 code"
  print("\nModel-5")
  print("2 Hidden layer with 2,3 Nodes")
  net = Net4().to(device)

  # 3. train network
  net = net.train()  # set training mode
  lrn_rate = 0.1

  loss_obj = T.nn.MSELoss()  # Mse loss function 
  optimizer = T.optim.SGD(net.parameters(),
    lr=lrn_rate)
  max_epochs = 100
  ep_log_interval = 10
  print("Loss function: " + str(loss_obj))
  print("Optimizer: SGD")
  print("Best Learn rate: 0.1")
  print("Batch size: 10")
  print("Max epochs: " + str(max_epochs))

  print("\nStarting training")
  for epoch in range(0, max_epochs):
    epoch_loss = 0.0            # for one full epoch
    

    for (batch_idx, batch) in enumerate(train_ldr):
      X = batch['predictors']  # inputs
      Y = batch['target']      #   targets
      oupt = net(X)            #   computeds 

      loss_val = loss_obj(oupt, Y)   # a tensor
      epoch_loss += loss_val.item()  # accumulate

      optimizer.zero_grad() # reset all gradients
      loss_val.backward()   # compute all gradients
      optimizer.step()      # update all weights

    if epoch % ep_log_interval == 0:
      print("epoch = %4d   loss = %0.4f" % \
        (epoch, epoch_loss))
      # print("custom loss = %0.4f" % epoch_loss_custom)
      # print("")
  print("Done ")

# ----------------------------------------------------------

  # 4. evaluate model
  net = net.eval()
  acc_train = accuracy(net, train_ds)
  print("\nAccuracy on train data = %0.2f%%" % \
    (acc_train * 100))
  acc_test = accuracy(net, test_ds)
  print("Accuracy on test data = %0.2f%%" % \
    (acc_test * 100))        

#ploting the scatter plot of reduced dimentional data 
  x1 = x[:, 0]
  x2 = x[:, 1]

  plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 2)
    )
  plt.title("Reduced Dimentional data in 2D plane")
  plt.xlabel("Principal Component 1")
  plt.ylabel("Principal Component 2")
  plt.colorbar()
  plt.show()

if __name__== "__main__":
  main()
