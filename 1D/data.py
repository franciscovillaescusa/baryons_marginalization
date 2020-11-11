import numpy as np
import torch
import sys,os


# create the data to train/validate/test the network
def dataset(k, Nk, kpivot, batch_size, predict_D, fix_C_value=True):

    # A goes from  0.1 to 10
    # B goes from -1.0 to 0.0
    # D goes from -0.5 to 0.5
    
    # get the values of the parameters/labels
    A  = (9.9*np.random.random(batch_size)+0.1)
    A2 = (A-0.1)/9.9 #normalized value of A
    B  = -1.0 + 1.0*np.random.random(batch_size)
    D  = -0.5 + np.random.random(batch_size)
    if predict_D:  label = np.array([A2, B, D], dtype=np.float32)
    else:          label = np.array([A2, B],    dtype=np.float32)

    # compute Pk
    Pk = np.zeros((batch_size, k.shape[0]), dtype=np.float32)
    for i in range(batch_size):
        Pk[i] = A[i]*k**B[i]

    # get the hydro Pk part
    indexes = np.where(k>kpivot)[0]
    if len(indexes)>0:
        C = Pk[:,indexes[0]]/k[indexes[0]]**D
        if not(fix_C_value):
            C = C*(0.5 + np.random.random(batch_size)*1.0)
        for i in range(batch_size):
            Pk[i,indexes] = C[i]*k[indexes]**D[i]

    # add cosmic variance
    dPk = np.sqrt(2*Pk**2/Nk)
    Pk  = np.random.normal(loc=Pk, scale=dPk)

    # save data to make plots
    #Pk_plot = np.zeros((batch_size+1,k.shape[0]), dtype=np.float32)
    #Pk_plot[0]  = k
    #Pk_plot[1:] = Pk
    #np.savetxt('borrar.txt', np.transpose(Pk_plot))
    
    # return data
    data = np.log10(Pk, dtype=np.float32) #Pk
    return torch.tensor(data), torch.tensor(label).t()


# create a dataset with fixed values of A and B to test the model and compare against
# Fisher matrix
def dataset_fid(k, Nk, kpivot, batch_size, A_fid, B_fid, predict_D, fix_C_value=True):

    
    # get the values of the parameters/labels
    A  = np.ones(batch_size)*A_fid
    A2 = (A-0.1)/9.9 #normalized value of A
    B  = np.ones(batch_size)*B_fid
    D  = -0.5 + np.random.random(batch_size)
    if predict_D:  label = np.array([A2, B, D], dtype=np.float32)
    else:          label = np.array([A2, B],    dtype=np.float32)

    # compute Pk
    Pk = np.zeros((batch_size, k.shape[0]), dtype=np.float32)
    for i in range(batch_size):
        Pk[i] = A[i]*k**B[i]

    # get the hydro Pk part
    indexes = np.where(k>kpivot)[0]
    if len(indexes)>0:
        C = Pk[:,indexes[0]]/k[indexes[0]]**D
        if not(fix_C_value):
            C = C*(0.8 + np.random.random(batch_size)*0.4)
        for i in range(batch_size):
            Pk[i,indexes] = C[i]*k[indexes]**D[i]

    # add cosmic variance
    dPk = np.sqrt(2*Pk**2/Nk)
    Pk  = np.random.normal(loc=Pk, scale=dPk)
    
    # return data
    data = np.log10(Pk, dtype=np.float32) #Pk
    return torch.tensor(data), torch.tensor(label).t()
