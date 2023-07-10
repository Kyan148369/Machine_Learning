import numpy as np
def perceptron_margin_classifier(data,labels,T):
    #intitalize th and th0 to 0 everytime when starting an iteration 
    #over here im making theta opposite cause i want to transpose and then multiply it by x for their dimensions to match
    #so i'm will append th0 too vertically along axis = 0 cause after transpose its effectively the same)
    th0 = np.array([[0]])
    array_1 = np.array([1])   
    th = np.zeros((data.shape[1]+1, 1))
    for t in range (T):
        #initialize thetas over here
    #if labels is 2 dimensional
        #if labels is 1 dimensional
        margin = []

        for i in range(len(data)):
            x_old  = np.array(data[i])
            x = np.concatenate((x_old,array_1.reshape(1,-1)),axis = 1)
            y = labels[i]
            margin.append( y*(np.dot(np.transpose(th),x))/np.linalg.norm(th))

            if y*(np.dot(np.transpose(th),x)) <= 0:  
                th = th + y*x
        margin = min(margin)

    return th,th0,margin


data = np.array([[200, 800, 200, 800], [0.2,  0.2,  0.8,  0.8]]).T
labels = np.array([-1, -1, 1, 1])

th, th0, margin = perceptron_margin_classifier(data, labels, 1)