# Ref : https://www.lptmc.jussieu.fr/user/talbot/sudoku.html ( metropolis method)
import numpy as np
import math 
import random
import matplotlib.pyplot as plt

# Algorithm Params
temp = 0.10;     
ntrial = 1000000;
emin = 18;
zero = 0;

# Functions used to compute energy

def check(i, k, ncheck):
# determines number of unique elements in each row (k=1) or column (k!=1)
    nu=0
    if k!=1:
        ncheck=np.transpose(ncheck)   
    nu=len(np.unique(ncheck[i,]))
    return(nu)

def checksq(Is, Js, ncheck):
    nu=0
    sCell=int(pow(ncheck.size,1/4)) # compute these kind of variable outsite
    subcell=ncheck[sCell*Is:sCell*Is+sCell,sCell*Js:sCell*Js+sCell]
    nu=len(np.unique(subcell))
    return(nu)

def energy(ncheck):
    nsum=0
    nCell=int(pow(ncheck.size,1/4))
    nmax=3*pow(nCell,4)
    nRange=np.arange(ncheck.shape[1])
    cRange=np.arange(int(pow(ncheck.size,1/4)))
    for i in nRange:
        nsum += check(i,1,n) + check(i,2,n)
    for i in cRange:
        for j in cRange:
            nsum += checksq(i,j,n)
    return(nmax-nsum)
    
## Read the Cell

gameFile="sudoku.dat" 
n=np.fromfile(gameFile,dtype=int,sep=" ") 
#n=np.zeros(25*25) # only for test
size=int(math.sqrt(len(n)))
gameRange=np.arange(size)
cellSize=int(math.sqrt(size))
cellRange=np.arange(cellSize)
n=n.reshape(size,size)

## Initialise variables
nums=np.zeros(size)
num1=np.zeros(size)
ntemp=0
ep=0

mask=(n==0)*1

# Fill the Cell with resolved boxes

for ib in cellRange:
    for jb in cellRange:
        for k in gameRange:
            nums[k]=k+1
        for i in cellRange:
            for j in cellRange:
                i1 = ib*cellSize + i
                j1 = jb*cellSize + j
                if n[i1][j1] !=0:
                    ix = n[i1][j1]
                    nums[ix-1]=0
        iy = -1
        for k in gameRange:
            if nums[k]!=0:
                iy+=1
                num1[iy] = nums[k]
        kk=0
        for i in cellRange:
            for j in cellRange:
                i1 = ib*cellSize + i
                j1 = jb*cellSize + j            
                if n[i1][j1] ==0:
                    n[i1][j1]=num1[kk]
                    kk+=1

print(n)
e=energy(n) # To optimize
En=[]
# start Monte Carlo loop
for ll in np.arange(ntrial):
     En.append(e)
    #  pick at random a block and two moveable elements in the block
     ib = cellSize*(int)(cellSize*random.uniform(0,1))
     jb = cellSize*(int)(cellSize*random.uniform(0,1))
     
     while True: 
         i1 = (int)(cellSize*random.uniform(0,1))
         j1 = (int)(cellSize*random.uniform(0,1))
         if mask[ib+i1][jb+j1]==1:
            break
     while True: 
         i2 = (int)(cellSize*random.uniform(0,1))
         j2 = (int)(cellSize*random.uniform(0,1))
         if mask[ib+i2][jb+j2]==1:
           break
     # swap and compute the energy of the trial
     ntemp = n[ib+i1][jb+j1]
     n[ib+i1][jb+j1] = n[ib+i2][jb+j2]
     n[ib+i2][jb+j2] = ntemp
     
     ep=energy(n)  
     if ep<emin:
         print("Step ",ll," energy= ",ep)
     if ep==0: # Solution found
         break
    
     if math.exp((e-ep)/temp) > random.uniform(0,1): 
         e=ep
     else:
         ntemp=n[ib+i1][jb+j1]
         n[ib+i1][jb+j1]=n[ib+i2][jb+j2]
         n[ib+i2][jb+j2]=ntemp
        
if ep==0:
    print("Solution found : ")
    print(n)
    plt.plot(En)
else:
    print("No solution found after ",ntrial ," steps")
    plt.plot(En)
