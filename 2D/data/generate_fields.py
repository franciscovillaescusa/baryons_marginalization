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
grid              = 128
BoxSize           = 1000.0 #Mpc/h
Rayleigh_sampling = 1
threads           = 1
verbose           = False
MAS               = 'None'
initial_seed      = 4 #to generate values of A
num_maps          = 100000
#######################################################################################

# initialize random seed and get the values of the parameters and the seeds
# notice that all cpus will have the same values
np.random.seed(initial_seed)
A    = 0.8 + 0.4*np.random.rand(num_maps)
B    = -0.5
seed = np.arange(num_maps)
 
# define the matrix hosting all the maps and power spectra
maps_partial = np.zeros((num_maps, grid, grid), dtype=np.float32)
maps_total   = np.zeros((num_maps, grid, grid), dtype=np.float32)
Pk_partial   = np.zeros((num_maps, 90),         dtype=np.float32)
Pk_total     = np.zeros((num_maps, 90),         dtype=np.float32)

# get the k-bins
k_in = np.logspace(-4,1,500, dtype=np.float32)

# find the numbers that each cpu will work with
numbers = np.where(np.arange(num_maps)%nprocs==myrank)[0]

# do a loop over all the maps
for i in numbers:

    if i%10000==0:  print(i)

    # get the value of the Pk
    Pk_in = A[i]*k_in**B

    # generate density field
    maps_partial[i] = DFL.gaussian_field_2D(grid, k_in, Pk_in, Rayleigh_sampling, 
                                            seed[i], BoxSize, threads, verbose)

    # compute power spectrum
    Pk_partial[i] = PKL.Pk_plane(maps_partial[i], BoxSize, MAS, threads, verbose).Pk
    #np.savetxt('Pk1.txt', np.transpose([Pk.k, Pk.Pk, Pk.Nmodes]))
    #print(1.0/np.sqrt(np.sum(Pk.Nmodes)))

    # make some statistics
    #print(np.mean(maps[i]), np.min(maps[i]), np.max(maps[i]))

    # make image
    """
    fig=figure()
    ax1=fig.add_subplot(111) 

    cax = ax1.imshow(df,cmap=get_cmap('jet'),origin='lower',interpolation='spline36',
                     extent=[0, BoxSize, 0, BoxSize])
    #vmin=min_density,vmax=max_density)
    #norm = LogNorm(vmin=min_density,vmax=max_density))
    #cbar = fig.colorbar(cax, ax2, ax=ax1, ticks=[-1, 0, 1]) #in ax2 colorbar of ax1
    #cbar.set_label(r"$M_{\rm CSF}\/[h^{-1}M_\odot]$",fontsize=14,labelpad=-50)
    #cbar.ax.tick_params(labelsize=10)  #to change size of ticks

    savefig('image1.png', bbox_inches='tight')
    close(fig)
    """

comm.Reduce(maps_partial, maps_total, root=0)
comm.Reduce(Pk_partial, Pk_total,     root=0)
if myrank==0:

    # check that there are no empty maps/zero Pk
    for i in range(num_maps):
        if np.std(maps_total[i])==0.0:
            print(i)
        if np.any(Pk_total[i]==0):
            print(i)

    # save maps, A and Pk
    np.save('Gaussian_maps.npy',maps_total)
    np.save('Power_spectra.npy', Pk_total)
    np.save('A_values.npy', A)
