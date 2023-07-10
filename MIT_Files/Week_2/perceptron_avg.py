import numpy as np

def averaged_perceptron(data, labels, params={'T':100}, hook=None):
            # if T not in params, default to 100
            # Your implementation here
    T = params.get('T',100)
    n = len(data)
    d = len(data[0])
    
    ths = np.zeros(d)
    th0s = 0

    th = np.zeros(d)
    th0 = 0

    for t in range (T):
        for i in range (n):
            x = np.transpose(data[i])
            y = labels[0,i]
            if np.dot(y,(np.dot(np. transpose(th),x)))+th0 <= 0:
                 th = th + np.dot(y,x)
                 th0 = th0 + y
            ths = ths + th
            th0s = th0s + th0
    return np.divide(ths,(n*T)), np.divide(th0s,(n*T))

data = np.array([[3,4],[4,5],[5,6]])
labels = np.array([[1,1,-1]])
th,th0 = averaged_perceptron(data,labels)
print(th,th0)