import numpy as np 


#over here data each will be in column form or nx1 so th*x th will have to be 1xn 
#we append column of 1s to data and 0s to bias of th 
def perceptron_classifier(data,labels,T):
    th = np.zeros(data.shape[1]+1)
    for t in range(T):
        
        for i in range(data.shape[0]):
            x = np.append(data[i],1)
            y = labels[i]
            if y*(np.dot(th,x)) <=0:
                th = th + y*x
    print(th[:-1],th [-1])
    return th[:-1],th [-1]


data =   np.array([[2], [3], [4],  [5]])
labels = np.array([1, 1, -1, -1])
T = 15
th,th0 = perceptron_classifier(data,labels,T)



