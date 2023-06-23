import numpy as np 

def perceptron_classifier(th,th0,T):
        for i in range(1, len(x)):
            #checks if the classification is on the right side or wrong side of the plane + 
            if np.dot(y[i],(np.dot(np.transpose(th),x[i]))) + th0 <= 0:
                print(th,th0)
                th = th + y[i]*x[i]
                th0 = th0 + y[i]
        return th,th0


    
y = np.array([[-1],[1]])
x = np.array([[1,-2],[-1,-4]])
th = np.array([0,0])  
th0 = np.array([0])
th,th0 = perceptron_classifier(th,th0,2)
print(th,th0)
