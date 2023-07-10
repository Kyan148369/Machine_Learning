import numpy as np 
import math


#We need to take the data from the dataset and then write the code for running cross validation using perceptron
# function parameters we need are data, number of folds we want to cross train on, T variable probably,
#perceptron we'll intitalize to 0 anyways
# for better functionality probably write functions separately one for cross validation and one for 
#the perceptron so we can put other algorithms too in the cross validation
#n sized break up 
def cross_validation(data, n , T ,labels):
    #Creating arrays for train and test
    errors_final = []
    #checking if it divides it perfectly or not if it doesnt then we have to add another loop
    #splitting into n folds
    data_fold = np.array_split(data,n)
    labels_fold = np.array_split(labels,n)
    th = np.zeros((len(data[0]), 1))
    th0 = 0
    #this loop is basically for the ith fold checks which one we are isolating 
    for i in range (n):
        
    # this loop is basically for the folds appends the data to the approriate train and test  oh I'd have to do it for labels too ig 
        for j in range (n):
            #these basically append it to test if they are on that particular fold else we will append it to training data
            if j!=i:
                training_data = np.concatenate(data_fold[j])
                training_labels = np.concatenate(labels_fold[j])
            elif j == i:
                test_data = data_fold[j]
                training_labels = labels_fold[j]
        #th, theta from train data
        th,th0 = perceptron(training_data,training_labels)
        #now run it on the test data and see the error for that fold
        error = error_rate(th,th0,test_data,test_labels)
        #append that iteration error and then calculate the average error
        errors_final.append(error)
    
#return the mean of the errors
    return np.mean(errors_final) 


def perceptron(th,th0,data,label):
    for t in range (T= 100):
    #write something that picks something at random and then proceeds to choose one of the remaining
        for i in range(len(data)):
            x = data[:,i]
            y = labels[i]
            if (np.dot(y,np.dot(np.transpose(th),x) + th))<= 0:
                th += y*x
                th0 = th + y
                
    return th,th0
            

def error_rate(th,th0,data,label,T : {100}):
    test_predict = []
    for i in range(len(data)):
        x = data[:,i]
        y = labels[i]
        if (y*(np.dot(np.transpose(th),x) + th0))<= 0:
            test_predict.append[0]
        else:
            test_predict.append[1]

    return np.mean(np.array(test_predictions) != labels)