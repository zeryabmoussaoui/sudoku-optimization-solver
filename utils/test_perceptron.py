import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

x=np.arange(100)/25
x=x.reshape(-1,1) # reshape for MLP
y=np.sin(x)*np.exp(-2*x)

plt.plot(x,y)
mlp = MLPRegressor(hidden_layer_sizes=(100,50), solver="sgd")
mlp2 = MLPRegressor(hidden_layer_sizes=(100,50), solver ="adam",batch_size=10 ) # change solver for lgbd or adam

# online one
for i in np.random.permutation(np.arange(len(x))): #TODO: choose a random sample
    inputX = np.array(x[i]).reshape(-1,1)
    inputY = np.array(y[i]).reshape(-1,1)
    mlp.partial_fit(inputX, inputY)

# offline one ( no minibatch)    
mlp2.fit(x,y)    

y_raw=[]
for i in np.arange(len(x)):
    inputX = np.array(x[i]).reshape(-1,1)
    y_raw.append(mlp.predict(inputX))

y_pred = np.array(y_raw)
y_pred2 = mlp2.predict(x)

#plt.plot(x,y_pred2)
plt.plot(x,y_pred)




