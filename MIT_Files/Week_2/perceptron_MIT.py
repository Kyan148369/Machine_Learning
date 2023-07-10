import numpy as np

def perceptron(data, labels, params={}):
    # if T not in params, default to 100
    
    d,n = data.shape
    th = np.zeros((d,1))
    th0 = np.zeros([1,n])
    
    # Your implementation here
    for i in range (n):
        y = labels[0,i]
        x = data[:,i]
        if (labels[i]*np.dot(np.transpose(theta),data[:i])+th0 <= 0):
            th = th + x[:,i]*y[i]
            th0 = th0 +labels[i]
    return th,th0
d =5
n = 3
data = np.zeros((d, n))
d = len(data[0])
labels = np.array([1,-1,-1])
th,th0 = perceptron(data,labels)
            
        
        