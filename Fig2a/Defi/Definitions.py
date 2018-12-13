# coding:utf-8
# Definitions.py, this program is about some fundamental definitons used in the programs
import numpy as np

# Sample complexity
# For training samples
complexity_sample = 100000
# For testing samples
complexity_sample_test = 10000

# The name of the file used to store the x testing data
x_test_data_store = "../Data/x_test_data.txt"

# The name of the file used to store the y testing data
y_test_data_store = "../Data/y_test_data.txt"

# The name of the file used to store the x training data
x_data_store = "../Data/x_data.txt"

# The name of the file used to store the y training data
y_data_store = "../Data/y_data.txt"

# The name of the file used to store the weights
w_data_store = "../Data/w_data.txt"

# The parameter dimension_x means the dimension of x (for the single input value)
dimension_x = 10

# The parameter sigma of features
square_sigma_1 = 1.5

# The parameter sigma of noise
square_sigma_2 = 0.5

# The parameter mu of noise
mu = 0

# The unifrom parameters, such as np.random.uniform(-inf_uniform, sup_uniform)
inf_uniform = 1
sup_uniform = 1

# The normal distribution used to compute
def norm_dis(mu, sigma, x):
    morm_density = np.power(np.sqrt(2 * np.pi), -1) * np.power(np.e, -0.5 * np.power((x - 1), 2))
    return morm_density
