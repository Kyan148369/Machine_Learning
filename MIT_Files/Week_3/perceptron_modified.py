import numpy as np

# Function for classifying perceptron
def perceptron_margin_classifier(data, labels, T):
    # Appending a column of ones to the data to account for the bias term
    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)

    # Initialize the weights (including the bias) to zero
    th = np.zeros(data.shape[1])

    # Initialize the margin to infinity for comparing
    margin = np.inf

    for t in range(T):  # zip over here since we are iterating over both variables
        for x, y in zip(data, labels):
            # Compute the prediction

            if y * np.dot(th, x) <= 0:
                # Update the weights
                th = th + y * x
            else:
                # Compute the margin
                margin_i = y * np.dot(th, x) / np.linalg.norm(th)
                # Update the margin
                margin = min(margin, margin_i)

    # Return th except the last term since that's the bias term, bias term, and margin
    return th[:-1], th[-1], margin


# Self-explanatory
def margin_and_no_of_mistakes(data, labels, T):
    # Appending a column of ones to the data to account for the bias term
    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)
    th, th0, margin = perceptron_margin_classifier(data, labels, T)
    # Over here, it represents a boolean value (True or False) so we will sum up all the values
    mistakes = sum(y * (np.dot(th, x)) <= 0 for x, y in zip(data, labels))
    return mistakes, margin


# Original data
data = np.array([[200, 800, 200, 800], [0.2, 0.2, 0.8, 0.8]]).T
labels = np.array([-1, -1, 1, 1])

# 1A) Compute the margin of the original data
# Returns th, th0, margin
_, _, margin_original = perceptron_margin_classifier(data, labels, 1)

# Check each column and determine the maximum magnitude for R
# 1B) Compute the theoretical bound on the number of mistakes
R_original = np.max(np.linalg.norm(data, axis=1))
bound_original = (R_original / margin_original) ** 2

# Returns mistakes and margin
# 1C) Compute the number of mistakes made by the perceptron on the original data
mistakes_original, _ = margin_and_no_of_mistakes(data, labels, 1)

# th, th0, margin_scaled
# 1D) Scale both features by 0.001 and compute the margin
data_scaled = data * 0.001
_, _, margin_scaled = perceptron_margin_classifier(data_scaled, labels, 1)

# 1E) Compute the theoretical bound on the number of mistakes after scaling both features
R_scaled = np.max(np.linalg.norm(data_scaled, axis=1))
bound_scaled = (R_scaled / margin_scaled) ** 2

# 1F) Scale only the first feature by 0.001 and compute the margin
data_partial_scaled = data.copy()
data_partial_scaled[:, 0] *= 0.001
_, _, margin_partial_scaled = perceptron_margin_classifier(data_partial_scaled, labels, 1)

# 1G) Compute the theoretical bound on the number of mistakes after scaling the first feature
R_partial_scaled = np.max(np.linalg.norm(data_partial_scaled, axis=1))
bound_partial_scaled = (R_partial_scaled / margin_partial_scaled) ** 2

# 1H) Compute the number of mistakes made by the perceptron after scaling the first feature
mistakes_partial_scaled, _ = margin_and_no_of_mistakes(data_partial_scaled, labels, 1)

margin_original, bound_original, mistakes_original, margin_scaled, bound_scaled, margin_partial_scaled, bound_partial_scaled, mistakes_partial_scaled

