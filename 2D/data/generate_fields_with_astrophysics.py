# This script generates 2D Gaussian maps with
# P(k) = A*k^B for k<k_pivot
# & 
# P(k) = C*k^D for k>k_pivot
from mpi4py import MPI
import numpy as np
import sys,os
import density_field_library as DFL
import Pk_library as PKL

from pylab import *
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
rcParams["mathtext.fontset"]='cm'

###### MPI DEFINITIONS ###### 
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

####################################### INPUT #########################################
root_out          = '/mnt/ceph/users/camels/Results/Gaussian_fields/data'
grid              = 128
BoxSize           = 1000.0 #Mpc/h
Rayleigh_sampling = 1
threads           = 1
verbose           = False
MAS               = 'None'
initial_seed      = 4 #to generate values of A and D
num_maps          = 100000

k_pivot    = 0.3   #h/Mpc
continuous = False #whether A*k_pivot^B = C*k_pivot^D
#######################################################################################

# initialize random seed and get the values of the parameters and the seeds
# notice that all cpus will have the same values
np.random.seed(initial_seed)
A    = 0.8 + 0.4*np.random.rand(num_maps)
B    = -0.5
D    = -1.0 + 2.0*np.random.rand(num_maps)
C    = A*k_pivot**(B-D)
if not(continuous):
    C = C*(0.7 + 0.6*np.random.rand(num_maps))

seed = np.arange(num_maps)
 
# define the matrix hosting all the maps and power spectra
maps_partial = np.zeros((num_maps, grid, grid), dtype=np.float32)
maps_total   = np.zeros((num_maps, grid, grid), dtype=np.float32)
Pk_partial   = np.zeros((num_maps, 90),         dtype=np.float32)
Pk_total     = np.zeros((num_maps, 90),         dtype=np.float32)

# get the k-bins
k_in = np.logspace(-4,1,500, dtype=np.float32)
indexes_low  = np.where(k_in<=k_pivot)[0]
indexes_high = np.where(k_in>k_pivot)[0]

# find the numbers that each cpu will work with
numbers = np.where(np.arange(num_maps)%nprocs==myrank)[0]

# do a loop over all the maps
for i in numbers:

    if i%10000==0:  print(i)

    # get the value of the Pk
    Pk_in = np.zeros(k_in.shape[0], dtype=np.float32)
    Pk_in[indexes_low]  = A[i]*k_in[indexes_low]**B
    Pk_in[indexes_high] = C[i]*k_in[indexes_high]**D[i]

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
    if continuous:
        np.save('%s/Gaussian_maps_kpivot=%.1f.npy'%(root_out,k_pivot), maps_total)
        np.save('%s/Power_spectra_kpivot=%.1f.npy'%(root_out,k_pivot), Pk_total)
        np.save('%s/A_values_kpivot=%.1f.npy'%(root_out,k_pivot), A)
    else:
        np.save('%s/Gaussian_maps_kpivot=%.1f_discon.npy'%(root_out,k_pivot), maps_total)
        np.save('%s/Power_spectra_kpivot=%.1f_discon.npy'%(root_out,k_pivot), Pk_total)
        np.save('%s/A_values_kpivot=%.1f_discon.npy'%(root_out,k_pivot), A)
