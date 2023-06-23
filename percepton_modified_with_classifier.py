import numpy as np 

# this function is a modified perceptron classifier which has the bias term appended to it
#but increaes the dimension 
def percepton_classifier(th,T):
    for i in range (len(x_append)):
        if (np.dot(y[i],np.dot(np.transpose(th),x_append[i]))<= 0):
            th = th + y[i]*x_append[i]
    return th

#this function calculates distance and then states on which side of the classifier is it on 
#margin = y[i]*(th T *x + th0)/ norm(th))
def calculate_margin(th):
    margin = 0
    for i in range(len(x_append)):
        margin = margin + (y[i] * (np.dot(np.transpose(th), x_append[i]) + th[-1])) / np.linalg.norm(th)
    return margin


th = np.array([0,0,0])
y = np.array([1,-1])
x = np.array([[2,2],[4,-4]])
col_1s = np.array([1,1]).reshape(-1,1)
x_append = np.append(x,col_1s,axis = 1)
th = percepton_classifier(th, 2)
print(th)

margin = calculate_margin(th)
print(margin)