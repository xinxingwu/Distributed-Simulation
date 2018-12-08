# coding:utf-8
# GenerationSamples.py, this program is used to generate the training and testing samples
import numpy as np
import Defi.Definitions as defi

# The parameter sigma of features
square_sigma_1 = defi.square_sigma_1

# The parameter sigma of noise
square_sigma_2 = defi.square_sigma_2

# The parameter mu of noise
mu = defi.mu

# The unifrom parameters
inf_uniform = defi.inf_uniform
sup_uniform = defi.sup_uniform

# Using to generate weights, the number of weights is 3 times of the dimension of the feature
w = np.random.randint(1, 10, [pow(defi.dimension_x, 3), 1])
np.savetxt(defi.w_data_store, w)

# Once the weights generated, it should been fixed. So it can load the weights has been generated in the above procedure
# w = np.loadtxt(defi.w_data_store).astype(np.float32)

# Train data
# The first five features follow the uniform distribution
x_part_1 = np.random.uniform(-inf_uniform, sup_uniform, (defi.complexity_sample, defi.dimension_x / 2))

# The first five features follow the normal distribution
x_part_2 = square_sigma_1 * np.random.randn(defi.complexity_sample, defi.dimension_x / 2)
x_data = np.concatenate((x_part_1, x_part_2), axis=1)
np.savetxt(defi.x_data_store, np.float32(x_data))
# print x_data.shape

y_data_init = []

for i in range(defi.dimension_x):
    for j in range(defi.dimension_x):
        for k in range(defi.dimension_x):
            y_data_init.append(x_data[:, i] * x_data[:, j] * x_data[:, k])

y_data = np.dot(np.array(y_data_init).transpose(), w) + np.random.normal(mu, square_sigma_2)
np.savetxt(defi.y_data_store, np.float32(y_data))
# print y_data.shape

# Test data
x_part_test_1 = np.random.uniform(-inf_uniform, sup_uniform, (defi.complexity_sample_test, defi.dimension_x / 2))
x_part_test_2 = square_sigma_1 * np.random.randn(defi.complexity_sample_test, defi.dimension_x / 2)
x_test_data = np.concatenate((x_part_test_1, x_part_test_2), axis=1)
np.savetxt(defi.x_test_data_store, np.float32(x_test_data))
# print x_test_data.shape

y_test_data_init = []

for i in range(defi.dimension_x):
    for j in range(defi.dimension_x):
        for k in range(defi.dimension_x):
            y_test_data_init.append(x_test_data[:, i] * x_test_data[:, j] * x_test_data[:, k])

y_test_data = np.dot(np.array(y_test_data_init).transpose(), w) + np.random.normal(mu, square_sigma_2)
np.savetxt(defi.y_test_data_store, np.float32(y_test_data))
# print y_test_data.shape
