import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys,os
import data as data
import architecture as architecture

####################################### INPUT ####################################
# k-values
kmin  = 7e-3 #h/Mpc
kmaxs = [0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #h/Mpc

# model parameters
kpivot        = 2.0
predict_D     = False
Pk_continuous = False
suffix        = 'Pk-60-60-60-60-2'

# architecture parameters
model   = 'model4' #'model1', 'model0'
hidden1 = 60 #for model0
hidden2 = 60 #for model0
hidden3 = 60 #for model0
hidden4 = 60 #for model0
hidden5 = 60 #for model0
hidden  = 500  #for model1
dr      = 0.0

A_fid = 5.0
B_fid = -0.5

# training parameters
batch_size_test  = 70000

fout = 'errors_%s.txt'%suffix
##################################################################################

# use GPUs if available
GPU    = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print('GPU: %s  ||  Training on %s'%(GPU, device))

# verbose
if predict_D:  print('kmax\t   dA   \t   dB   \t   dC')
else:          print('kmax\t   dA   \t   dB')

# find suffix
suffix = '%s_kpivot=%.2f'%(suffix,kpivot)
if not(Pk_continuous):  suffix = '%s_discon'%suffix

f = open(fout, 'w')
for kmax in kmaxs[::-1]:

    # find the fundamental frequency, the number of bins up to kmax and the k-array
    kF     = kmin
    k_bins = int((kmax-kmin)/kF)
    k      = np.arange(3,k_bins+2)*kF #avoid k=kF as we will get some negative values
    Nk     = 4.0*np.pi*k**2*kF/kF**3  #number of modes in each k-bin

    # find the number of neurons in the output layer and define loss
    if predict_D:  last_layer = 3
    else:          last_layer = 2

    # get the test dataset
    test_data, test_label = data.dataset_fid(k, Nk, kpivot, batch_size_test, 
                                             A_fid, B_fid, predict_D, Pk_continuous)
    #test_data, test_label = data.dataset(k, Nk, kpivot, batch_size_test, 
    #                                     predict_D, Pk_continuous)

    # get the architecture
    if   model=='model0':
        net = architecture.Model(k.shape[0], hidden1, hidden2, hidden3, hidden4, 
                                 hidden5, last_layer).to(device)
    #elif model=='model3':
    #    net = architecture.Model1(k.shape[0], hidden, last_layer).to(device)
    elif model=='model1':
        net = architecture.model_1hl(k.shape[0], hidden1, last_layer, dr).to(device)
    elif model=='model2':
        net = architecture.model_2hl(k.shape[0], hidden1, hidden2, last_layer, 
                                     dr).to(device)
    elif model=='model3':
        net = architecture.model_3hl(k.shape[0], hidden1, hidden2, hidden3, last_layer, 
                                     dr).to(device)
    elif model=='model4':
        net = architecture.model_4hl(k.shape[0], hidden1, hidden2, hidden3, hidden4,
                                     last_layer, dr).to(device)
    else:  raise Exception('Wrong model!')

    # load best model
    fmodel = 'GPU_results/models/best_model_%s_kmax=%.2f.pt'\
             %(suffix,kmax)
    net.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))

    # get NN prediction
    net.eval()
    with torch.no_grad():
        pred = net(test_data)

    A_pred, B_pred = pred[:,0]*9.9 + 0.1,       pred[:,1]
    A_test, B_test = test_label[:,0]*9.9 + 0.1, test_label[:,1]

    np.savetxt('borrar.txt', np.transpose([B_pred.numpy(), B_test.numpy()]))
    print('A = %.6f +- %.6f'%(np.mean(A_pred.numpy(), dtype=np.float64),
                              np.std(A_pred.numpy(), dtype=np.float64)))
    print('B = %.6f +- %.6f'%(np.mean(B_pred.numpy(), dtype=np.float64),
                              np.std(B_pred.numpy(), dtype=np.float64)))

    dA = np.sqrt(np.mean(((A_pred - A_test)**2).numpy()))
    dB = np.sqrt(np.mean(((B_pred - B_test)**2).numpy()))
        
    if predict_D:
        dC = np.sqrt(np.mean(((pred[:,2]-test_label[:,2])**2).numpy()))
        print('%.2f\t%.3e\t%.3e\t%.3e'%(kmax,dA,dB,dC))
        f.write('%.2f %.4e %.4e %.4e\n'%(kmax,dA,dB,dC))
    else:
        print('%.2f\t%.3e\t%.3e'%(kmax,dA,dB))
        f.write('%.2f %.4e %.4e\n'%(kmax,dA,dB))

    sys.exit()
f.close()
