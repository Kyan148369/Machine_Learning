import numpy as np

def perceptron_margin_classifier(data, labels, T):
    d = data.shape[1]  # number of features
    th = np.zeros(d+1)  # initialize theta (including theta_0)
    margin = np.inf  # initialize margin to be very large

    for t in range(T):
        for i in range(len(data)):
            x_old = data[i]
            x = np.concatenate((x_old, [1]))  # append 1 to the data vector
            y = labels[i]

            if y * np.dot(th, x) <= 0:  # mistake is made
                th = th + y * x  # update theta
            else:  # correctly classified, update margin if necessary
                margin_i = y * np.dot(th, x) / np.linalg.norm(th)
                margin = min(margin, margin_i)
    return th, margin

# Testing the function
data = np.array([[200, 800, 200, 800], [0.2, 0.2, 0.8, 0.8]]).T
labels = np.array([-1, -1, 1, 1])

th, margin = perceptron_margin_classifier(data, labels, 1)

print("Theta:", th)
print("Margin:", margin)
