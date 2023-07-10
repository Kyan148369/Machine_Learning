import numpy as np 

def perceptron_classifier(th,th0,T,tol):
        mistakes_overall = 0 
        for i in range(1, len(x))
                mistakes_in_lastcycle_ = len(x)
                while (mistakes_in_lastcycle != 0):
                #checks if the classification is on the right side or wrong side of the plane + 
                        if np.dot(y[i],(np.dot(np.transpose(th),x[i]))) + th0 <= 0:
                                mistakes_overall = 0 
                                th = th + y[i]*x[i]
                                th0 = th0 + y[i]
                        else:
                                mistakes_in_lastcycle = mistakes_in_lastcycle-1
        return th,th0,mistakes_in_lastcycle,mistakes_overall
                         



    
y = np.array([[-1],[1]])
x = np.array([[1,-2],[-1,-4]])
th = np.array([0,0])  
th0 = np.array([0])
th,th0 = perceptron_classifier(th,th0,2,0.01)
print(th,th0)
