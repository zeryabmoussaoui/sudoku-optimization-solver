import numpy as np
import math
import random
import matplotlib.pyplot as plt
# ref : http://outlace.com/rlpart3.html
# TODO : add ploting method 


class QAgent:
     # TODO : inherit from some RL classes which implement alpha , gamma , etc and delete them from constructor 
     # TODO : make growing strategy parameter
     # TODO : add a param for constant learning rate
     decr = 0.9999 # decrease factor param (tune param)
         
     def __init__(self, env, interp, alpha = 1, gamma = 0.5, eps = 1, maxIter = 10, maxAction = 50000, nbStateBatch = 10000 ):
         self.env = env
         self.interp = interp
         self.interp.env = self.env
         self.nbAction = self.env.actions.shape[0]
         self.nbStateBatch = nbStateBatch # to grow qvalue table
         self.qValue = np.zeros((nbStateBatch, self.nbAction))

         self.alpha = alpha 
         self.gamma = gamma  
         self.eps = eps 
         self.maxIter = maxIter
         self.maxAction = maxAction
               
     def updateQ(self, prevState, prevAction, currentState, reward, alphaI):          
          oldInfo = (1-alphaI) * self.qValue[prevState, prevAction] 
          newInfo = alphaI * ( reward + self.gamma * np.max(self.qValue[currentState,:]) )
          self.qValue[prevState, prevAction] = oldInfo + newInfo  

     def chooseAction(self,fromState,epsI):  # TODO : uniform with QNN one
         epsThreshold = random.uniform(0,1)
         if epsThreshold  > epsI:
             # Exploitaton
             choosedAction = np.argmax(self.qValue[fromState,:])
         else:
             # Exploration
             choosedAction = random.randint(0, self.nbAction - 1)
         return(choosedAction)   
     
     def process(self):
        print("QL processing ...")
        j = 0
        while j < self.maxIter:
            i = 0
            newEnv = self.env.getInstance()
            self.env = newEnv
            currentState = self.interp.getState(self.env.observation)
            alphaI = self.alpha
            epsI = self.eps
            while i < self.maxAction:
                prevState = currentState
                selectedAction = self.chooseAction(currentState,epsI)
                reward, currentObs= self.env.applyAction(selectedAction)
                prevState = currentState
                currentState = self.interp.getState(currentObs)
                self.updateQ(prevState, selectedAction, currentState, reward, alphaI)
                alphaI = alphaI * self.decr
                epsI = epsI * self.decr
                if currentState == self.qValue.shape[0]-1 : 
                    print("Growing Q ..") # TODO : put in method
                    growTable = np.zeros((self.qValue.shape[0] + self.nbStateBatch, self.qValue.shape[1]))
                    growTable[:self.qValue.shape[0], :self.qValue.shape[1] ] = self.qValue
                    self.qValue = growTable
                if self.env.isFinalState == 1:
                    print("Final state ! at epoch : ",i)
                    print(self.env.observation)
                    break
                i = i + 1
            j = j + 1

            
class EnvSudoku:
     def __init__(self):
        gameFile="sudoku.dat" # TODO : manage mask
        n=np.fromfile(gameFile,dtype=int,sep=" ") 
        size=int(math.sqrt(len(n)))
        gameRange=np.arange(size)
        cellSize=int(math.sqrt(size))
        cellRange=np.arange(cellSize)
        n=n.reshape(size,size)
        mask=(n==0)*1
        ## Initialise Observation ( board)
        nums=np.zeros(size)
        num1=np.zeros(size)
        for ib in cellRange: # fill in the gaps with determinist local cells resolution
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

        self.observation = n  # The board is observed
        
        # Generate actions , these ones are specific for the input file and cannot be generalize
        forbidenActions = np.argwhere(mask.reshape(-1) == 0 ) # non movable cells , from file

        rawActions =  np.ones(n.size) # all permutations between 2 cells
        rawActions =  np.triu(rawActions) # Delete symetric permutation 
        rawActions =  rawActions - np.eye(n.size) # Delete unitary permutation 
        rawActions[:,forbidenActions] = 0
        rawActions[forbidenActions,] = 0
        rawActions = np.argwhere(rawActions == 1)
        
        realActions = np.zeros((len(rawActions),2,2))
        realActions[:,:,0] = rawActions // len(n)  # convert index to coordinates 
        realActions[:,:,1] = (rawActions) %  len(n)
        
        self.actions = realActions.astype(int) # List of useful permutations between two cells 
        self.isFinalState = 0 # TODO : to be moved on interpretor
        
     def applyAction(self, action):
         cellOne,cellTwo = self.actions[action,:]
         temp = self.observation[cellOne[0], cellOne[1]]
         self.observation[cellOne[0], cellOne[1]] = self.observation[cellTwo[0], cellTwo[1]]
         self.observation[cellTwo[0], cellTwo[1]] = temp
         reward = self.getReward()
         obs = self.observation
         return (reward , obs)
     
     def getReward(self):
        # TODO : try different reward like : alpha*delta(energy)
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
       
        nsum=0
        ncheck = self.observation
        nCell=int(pow(ncheck.size,1/4))
        nmax=3*pow(nCell,4)
        nRange=np.arange(ncheck.shape[1])
        cRange=np.arange(int(pow(ncheck.size,1/4)))
        for i in nRange:
            nsum += check(i,1,ncheck) + check(i,2,ncheck)
        for i in cRange:
            for j in cRange:
                nsum += checksq(i,j,ncheck)
        energy = nmax-nsum
        if energy != 0:
            reward = 100 * (1 / energy) # TODO : function from to tune 
            #reward = 10 * (1 / energy) # good value for tabular Q
        else:
            reward = 10000 # To tune
            #reward = 1000 # good value for tabular Q
            self.isFinalState = 1
        return(reward)
        
     def getInstance(self):  
        return (type(self)())
        

class Interpretor:
     def __init__(self, env):
         initObservation = env.observation
         self.stateList = []
         self.getState(initObservation) #only to init stateList
     def getState(self, obs): 
         rawState = obs.reshape(-1)
         n = rawState.shape[0] # size of board
         k =  int(math.sqrt(n)) # box size
         currentState = 0
         # convert state to number for compression
         for i in np.arange(rawState.shape[0]):
             currentState = currentState + rawState[i] * ( (k+1) ** i)  
             
         isSeenState = np.argwhere(self.stateList == currentState) # TODO init issue
         if isSeenState.any(): #known state
             foundState = isSeenState[0,0]
             state = foundState
         else: # discovered state
             self.stateList = np.append(self.stateList, currentState) 
             state =  self.stateList.size - 1 # id of the state
         return(state)
     def getStateVector(self,obs):
         # TODO : shall encode to mean someting
         currentObs = np.array(obs.reshape(1,-1)) # FIX , create new instance to avoid reference copy
         return(currentObs)

## Main

myEnv = EnvSudoku()
myInter = Interpretor(myEnv)

myQ = QAgent(myEnv,myInter)
myQ.process()
     
     
# Exploitation 
myQ.eps = 0.05 # 5% d'aléatoire
myQ.process()
    
    