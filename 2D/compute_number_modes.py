import numpy as np
import sys,os

grid = 128
k_pivot = 0.3
BoxSize = 1000.0

middle = grid//2
factor = 2.0*np.pi/BoxSize


modes = 0
for kxx in range(grid):
    kx = (kxx-grid if (kxx>middle) else kxx)
        
    for kyy in range(middle+1): #kyy=[0,1,..,middle] --> ky>0
        ky = (kyy-grid if (kyy>middle) else kyy)

        # ky=0 & ky=middle are special (modes with (kx<0, ky=0) are not
        # independent of (kx>0, ky=0): delta(-k)=delta*(+k))
        if ky==0 or (ky==middle and grid%2==0):
            if kx<0:  continue

        # compute |k| of the mode and its integer part
        k = np.sqrt(kx*kx + ky*ky)*factor
        if k<=k_pivot:  modes+=1

print('%d modes for k<%.2f'%(modes,k_pivot))
        
