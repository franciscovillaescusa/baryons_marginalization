# This script will generate thousands of Gaussian maps with A=1
from mpi4py import MPI
import numpy as np
import sys,os
import density_field_library as DFL
import Pk_library as PKL


###### MPI DEFINITIONS ###### 
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

####################################### INPUT #########################################
grid              = 128
BoxSize           = 1000.0 #Mpc/h
Rayleigh_sampling = 1
threads           = 1
verbose           = False
MAS               = 'None'
num_maps          = 100000
root_out          = '/mnt/ceph/users/camels/Results/Gaussian_fields/data'
A_value           = 1.1 #all maps will have this value of the A parameter
#######################################################################################

# initialize random seed and get the values of the parameters and the seeds
# notice that all cpus will have the same values
A    = np.ones(num_maps, dtype=np.float32)*A_value
B    = -0.5
seed = np.arange(456789, 456789+num_maps)
 
# define the matrix hosting all the maps and power spectra
maps_partial = np.zeros((num_maps, grid, grid), dtype=np.float32)
maps_total   = np.zeros((num_maps, grid, grid), dtype=np.float32)
Pk_partial   = np.zeros((num_maps, 90),         dtype=np.float32)
Pk_total     = np.zeros((num_maps, 90),         dtype=np.float32)

# get the k-bins and Pk
k_in  = np.logspace(-4,1,500, dtype=np.float32)

# find the numbers that each cpu will work with
numbers = np.where(np.arange(num_maps)%nprocs==myrank)[0]

# do a loop over all the maps
for i in numbers:

    if i%10000==0:  print(i)

    Pk_in = A[i]*k_in**B

    # generate density field
    maps_partial[i] = DFL.gaussian_field_2D(grid, k_in, Pk_in, Rayleigh_sampling, 
                                            seed[i], BoxSize, threads, verbose)

    # compute power spectrum
    Pk_partial[i] = PKL.Pk_plane(maps_partial[i], BoxSize, MAS, threads, verbose).Pk


comm.Reduce(maps_partial, maps_total, root=0)
comm.Reduce(Pk_partial, Pk_total,     root=0)
if myrank==0:

    # check that there are no empty maps/zero Pk
    for i in range(num_maps):
        if np.std(maps_total[i])==0.0:  print(i)
        if np.any(Pk_total[i]==0):      print(i)

    # save maps, A and Pk
    np.save('%s/Gaussian_maps_test_A=%.2f.npy'%(root_out, A_value), maps_total)
    np.save('%s/Power_spectra_test_A=%.2f.npy'%(root_out, A_value), Pk_total)
    np.save('%s/A_values_test_A=%.2f.npy'%(root_out, A_value),      A)
    
