
import numpy as np
import torch as T
import pprint
import math
import matplotlib
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
device = T.device("cpu")  # apply to Tensor or Module

"-------preprocessing of data------------"


class train_data(T.utils.data.Dataset):

    def __init__(self, src_file, num_rows=None):
        all_data = np.loadtxt(src_file, max_rows=num_rows,
                              usecols=range(1, 9), delimiter=",", skiprows=1,
                              dtype=np.float32)  # strip IDs off
        pprint.pprint(all_data[1])
        self.x_data = T.tensor(all_data[0:599, 0:7],
                               dtype=T.float32).to(device)
        pprint.pprint(self.x_data[1])
        self.y_data = T.tensor(all_data[0:599, 7],
                               dtype=T.float32).to(device)
        pprint.pprint(self.y_data[1])
        self.y_data = self.y_data.reshape(-1, 1)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()
        preds = self.x_data[idx, :]  # idx rows, all 4 cols
        lbl = self.y_data[idx, :]    # idx rows, the 1 col
        sample = {'predictors': preds, 'target': lbl}

        return sample


class test_data_set(T.utils.data.Dataset):

    def __init__(self, src_file, num_rows=None):
        all_data = np.loadtxt(src_file, max_rows=num_rows,
                              usecols=range(1, 9), delimiter=",", skiprows=1,
                              dtype=np.float32)  # strip IDs off

        self.x_data = T.tensor(all_data[600:750, 0:7],
                               dtype=T.float32).to(device)
        self.y_data = T.tensor(all_data[600:750, 7],
                               dtype=T.float32).to(device)

        # n_vals = len(self.y_data)
        # self.y_data = self.y_data.reshape(n_vals,1)
        self.y_data = self.y_data.reshape(-1, 1)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()
        preds = self.x_data[idx, :]  # idx rows, all 4 cols
        lbl = self.y_data[idx, :]    # idx rows, the 1 col
        sample = {'predictors': preds, 'target': lbl}
        # sample = dict()   # or sample = {}
        # sample["predictors"] = preds
        # sample["target"] = lbl

        return sample


# ---------------------------------------------------------
"---------------helper functions-----------------"


def accuracy(model, ds):
    # ds is a iterable Dataset of Tensors
    n_correct = 0
    n_wrong = 0

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
    num_correct = T.sum(targets == pred_y)
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
"-------------------Model implimentations---------------------"

# neural network function


class Net0(T.nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.oupt = T.nn.Linear(7, 1)

        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = T.sigmoid(self.oupt(x))
        return z


class Net1(T.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.hid1 = T.nn.Linear(7, 6)  # 7-6-1
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
        self.hid1 = T.nn.Linear(7, 2)
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
        self.hid1 = T.nn.Linear(7, 3)
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
        self.hid1 = T.nn.Linear(7, 2)  #
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
def plot_graph(a1, a2, a3, a4, a5):

    # GRAPH FOR LEARNING RATE VS ACCURACY
    # print(accList)
    X_N = [1.0, 2.0, 3.0, 4.0, 5.0]
    plt.plot(X_N, a1, 'black', label='MODEL 1')
    plt.plot(X_N, a2, 'orange', label='MODEL 2')
    plt.plot(X_N, a3, 'green', label='MODEL 3')
    plt.plot(X_N, a4, 'red', label='MODEL 4')
    plt.plot(X_N, a5, 'blue', label='MODEL 5')
    plt.title(' Learning Rate vs Accuracy ')
    plt.ylabel('Accuracies')
    plt.xlabel('Learning Rate ')
    plt.grid()
    plt.xticks([1.0, 2.0, 3.0, 4.0, 5.0], [
               "0.1", "0.01", "0.001", "0.001", "0.00001"])
    plt.legend()
    plt.show()
# Graph for model vs accuracy
    l1 = [a1[0], a2[0], a3[0], a4[0], a5[0]]
    l2 = [a1[1], a2[1], a3[1], a4[1], a5[1]]
    l3 = [a1[2], a2[2], a3[2], a4[2], a5[2]]
    l4 = [a1[3], a2[3], a3[3], a4[3], a5[3]]
    l5 = [a1[4], a2[4], a3[4], a4[4], a5[4]]
    Y_N = [1.0, 2.0, 3.0, 4.0, 5.0]
    plt.plot(Y_N, l1, 'black', label='LR 0.1')
    plt.plot(Y_N, l2, 'orange', label='LR 0.01')
    plt.plot(Y_N, l3, 'green', label='LR 0.001')
    plt.plot(Y_N, l4, 'red', label='LR 0.0001')
    plt.plot(Y_N, l5, 'blue', label='LR 0.00001')
    plt.title('Model vs Accuracy')
    plt.ylabel('Accuracies')
    plt.xlabel('Model')
    plt.grid()
    plt.xticks([1.0, 2.0, 3.0, 4.0, 5.0], ["Model 1",
               "Model 2", "Model 3", "Model 4", "Model 5"])
    plt.legend()
    plt.show()


"----------------main function----------------"


def main():
    # TO store accuracy of models
    accList1 = list([])
    accList2 = list([])
    accList3 = list([])
    accList4 = list([])
    accList5 = list([])

    # 0. get started
    print("\nDiabetes classification problem using PyTorch \n")
    T.manual_seed(1)
    np.random.seed(1)

    train_file = "diabetes.csv"
    test_file = "diabetes.csv"

    train_ds = train_data(train_file)  # all rows


# get first sample and unpack

    test_ds = test_data_set(test_file)

    bat_size = 10
    train_ldr = T.utils.data.DataLoader(train_ds,
                                        batch_size=bat_size, shuffle=True)
    # test_ldr = T.utils.data.DataLoader(test_ds,
    #   batch_size=1, shuffle=False)  # not used
    print("--------------MODELS PARAMETERS-------------")
    print("Loss function: MSE")
    print("Optimizer: SGD")
    print("Learn rate: VARIABLE")
    print("Batch size: 10")
    print("Max epochs: 100")
    # 2. create neural network
    first_data = train_ds[2]
    features, labels = first_data
    print("\n-----------------Model 1----------------\n")
    print(" ZERO Hidden layer\n")
    lrate = list([0.1, 0.01, 0.001, 0.0001, 0.00001])
    for l in lrate:
        net = Net0().to(device)

        print("\n FOR LEARNING RATE : %0.5f\n" % (l))
    # 3. train network
        #print("\nPreparing training")
        net = net.train()  # set training mode
        lrn_rate = l
        loss_obj = T.nn.MSELoss()  # MSE loss function
        optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
        max_epochs = 100
        ep_log_interval = 10

        print("\nStarting training")
        for epoch in range(0, max_epochs):
            epoch_loss = 0.0            # for one full epoch
            epoch_loss_custom = 0.0
            num_lines_read = 0

            for (batch_idx, batch) in enumerate(train_ldr):
                X = batch['predictors']  # [10,4]  inputs
                Y = batch['target']      # [10,1]  targets
                oupt = net(X)            # [10,1]  computeds

                loss_val = loss_obj(oupt, Y)   # a tensor
                epoch_loss += loss_val.item()  # accumulate
        # epoch_loss += loss_val  # is OK
        # epoch_loss_custom += my_bce(net, batch)

                optimizer.zero_grad()  # reset all gradients
                loss_val.backward()   # compute all gradients
                optimizer.step()      # update all weights

            if epoch % ep_log_interval == 0:
                print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))


# ----------------------------------------------------------

    # 4. evaluate model
        net = net.eval()
        acc_train = accuracy(net, train_ds)
        print("\nAccuracy on train data = %0.2f%%" %
              (acc_train * 100))
        acc_test = accuracy(net, test_ds)
        print("Accuracy on test data = %0.2f%%" %
              (acc_test * 100))
        accList1.append(acc_test * 100)

    print("\n-----------------Model 2----------------\n")
    print("1 Hidden Layer With 6 Nodes")
    for l in lrate:
        net = Net1().to(device)
        print("\nFOR LEARNING RATE : %0.5f\n" % (l))
    # 3. train network
        #print("\nPreparing training")
        net = net.train()  # set training mode
        lrn_rate = l
        loss_obj = T.nn.MSELoss()  # binary cross entropy
        optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
        max_epochs = 100
        ep_log_interval = 10
        print("\nStarting training")
        for epoch in range(0, max_epochs):
            epoch_loss = 0.0            # for one full epoch
            epoch_loss_custom = 0.0
            num_lines_read = 0

            for (batch_idx, batch) in enumerate(train_ldr):
                X = batch['predictors']  # [10,4]  inputs
                Y = batch['target']      # [10,1]  targets
                oupt = net(X)            # [10,1]  computeds

                loss_val = loss_obj(oupt, Y)   # a tensor
                epoch_loss += loss_val.item()  # accumulate
        # epoch_loss += loss_val  # is OK
        # epoch_loss_custom += my_bce(net, batch)

                optimizer.zero_grad()  # reset all gradients
                loss_val.backward()   # compute all gradients
                optimizer.step()      # update all weights

            if epoch % ep_log_interval == 0:
                print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))

        print("Done ")

# ----------------------------------------------------------

    # 4. evaluate model
        net = net.eval()
        acc_train = accuracy(net, train_ds)
        print("\nAccuracy on train data = %0.2f%%" %
              (acc_train * 100))
        acc_test = accuracy(net, test_ds)
        print("Accuracy on test data = %0.2f%%" %
              (acc_test * 100))
        accList2.append(acc_test * 100)

    print("\n-----------------Model 3----------------\n")
    print("1 Hidden Layer With 2 Nodes")
    for l in lrate:
        net = Net2().to(device)
        print("\nFOR LEARNING RATE : %0.5f\n" % (l))
    # 3. train network
        #print("\nPreparing training")
        net = net.train()  # set training mode
        lrn_rate = l
        loss_obj = T.nn.MSELoss()  # binary cross entropy
        optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
        max_epochs = 100
        ep_log_interval = 10
        print("\nStarting training")
        for epoch in range(0, max_epochs):
            epoch_loss = 0.0            # for one full epoch

            for (batch_idx, batch) in enumerate(train_ldr):
                X = batch['predictors']  # [10,4]  inputs
                Y = batch['target']      # [10,1]  targets
                oupt = net(X)            # [10,1]  computeds

                loss_val = loss_obj(oupt, Y)   # a tensor
                epoch_loss += loss_val.item()  # accumulate

                optimizer.zero_grad()  # reset all gradients
                loss_val.backward()   # compute all gradients
                optimizer.step()      # update all weights

            if epoch % ep_log_interval == 0:
                print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))

        print("Done ")

# ----------------------------------------------------------

    # 4. evaluate model
        net = net.eval()
        acc_train = accuracy(net, train_ds)
        print("\nAccuracy on train data = %0.2f%%" %
              (acc_train * 100))
        acc_test = accuracy(net, test_ds)
        print("Accuracy on test data = %0.2f%%" %
              (acc_test * 100))
        accList3.append(acc_test * 100)

    print("\n-----------------Model 4----------------\n")
    print("2 Hidden Layer With 3,2 Nodes")
    for l in lrate:
        net = Net3().to(device)
        print("\nFOR LEARNING RATE : %0.5f\n" % (l))
    # 3. train network
        #print("\nPreparing training")
        net = net.train()  # set training mode
        lrn_rate = l
        loss_obj = T.nn.MSELoss()  # MEAN SQUARE ERROR
        optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
        max_epochs = 100
        ep_log_interval = 10
        print("\nStarting training")
        for epoch in range(0, max_epochs):
            epoch_loss = 0.0            # for one full epoch
            epoch_loss_custom = 0.0
            num_lines_read = 0

            for (batch_idx, batch) in enumerate(train_ldr):
                X = batch['predictors']  # [10,4]  inputs
                Y = batch['target']      # [10,1]  targets
                oupt = net(X)            # [10,1]  computeds

                loss_val = loss_obj(oupt, Y)   # a tensor
                epoch_loss += loss_val.item()  # accumulate
        # epoch_loss += loss_val  # is OK
        # epoch_loss_custom += my_bce(net, batch)

                optimizer.zero_grad()  # reset all gradients
                loss_val.backward()   # compute all gradients
                optimizer.step()      # update all weights

            if epoch % ep_log_interval == 0:
                print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))

        print("Done ")

# ----------------------------------------------------------

    # 4. evaluate model
        net = net.eval()
        acc_train = accuracy(net, train_ds)
        print("\nAccuracy on train data = %0.2f%%" %
              (acc_train * 100))
        acc_test = accuracy(net, test_ds)
        print("Accuracy on test data = %0.2f%%" %
              (acc_test * 100))
        accList4.append(acc_test * 100)

    print("\n-----------------MOdel 5----------------\n")
    print("2 Hidden Layer With 2,3 Nodes")
    for l in lrate:
        net = Net4().to(device)
        print("\nFOR LEARNING RATE : %0.5f\n" % (l))
    # 3. train network
        #print("\nPreparing training")
        net = net.train()  # set training mode
        lrn_rate = l
        loss_obj = T.nn.MSELoss()  # binary cross entropy
        optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
        max_epochs = 100
        ep_log_interval = 10
        print("\nStarting training")
        for epoch in range(0, max_epochs):
            epoch_loss = 0.0            # for one full epoch
            epoch_loss_custom = 0.0
            num_lines_read = 0

            for (batch_idx, batch) in enumerate(train_ldr):
                X = batch['predictors']  # [10,4]  inputs
                Y = batch['target']      # [10,1]  targets
                oupt = net(X)            # [10,1]  computeds

                loss_val = loss_obj(oupt, Y)   # a tensor
                epoch_loss += loss_val.item()  # accumulate
        # epoch_loss += loss_val  # is OK
        # epoch_loss_custom += my_bce(net, batch)

                optimizer.zero_grad()  # reset all gradients
                loss_val.backward()   # compute all gradients
                optimizer.step()      # update all weights

            if epoch % ep_log_interval == 0:
                print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))

        print("Done ")

# ----------------------------------------------------------

    # 4. evaluate model
        net = net.eval()
        acc_train = accuracy(net, train_ds)
        print("\nAccuracy on train data = %0.2f%%" %
              (acc_train * 100))
        acc_test = accuracy(net, test_ds)
        print("Accuracy on test data = %0.2f%%" %
              (acc_test * 100))
        accList5.append(acc_test * 100)

    print("\n-----------------Best Model----------------\n")
    print("1 Hidden Layer With 6 Nodes")

    net = Net1().to(device)

    # 3. train network
    #print("\nPreparing training")
    net = net.train()  # set training mode
    lrn_rate = 0.1
    loss_obj = T.nn.MSELoss()  # MSE LOSS FUNCTION
    optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
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
            X = batch['predictors']  # [10,7]  inputs
            Y = batch['target']      # [10,1]  targets
            oupt = net(X)  # computeds

            loss_val = loss_obj(oupt, Y)   # a tensor
            epoch_loss += loss_val.item()  # accumulate
        # epoch_loss += loss_val  # is OK
        # epoch_loss_custom += my_bce(net, batch)

            optimizer.zero_grad()  # reset all gradients
            loss_val.backward()   # compute all gradients
            optimizer.step()      # update all weights

        if epoch % ep_log_interval == 0:
            print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))

    print("Done ")

# ----------------------------------------------------------

    # 4. evaluate model
    net = net.eval()
    acc_train = accuracy(net, train_ds)
    print("\nAccuracy on train data = %0.2f%%" %
          (acc_train * 100))
    acc_test = accuracy(net, test_ds)
    print("Accuracy on test data = %0.2f%%" %
          (acc_test * 100))

    plot_graph(accList1, accList2, accList3, accList4, accList5)


if __name__ == "__main__":
    main()
