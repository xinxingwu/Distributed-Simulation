# coding:utf-8
# This program is used to do experiment on risks/errors change with the regularization parameter λ
import tensorflow as tf
import numpy as np
import Defi.Definitions as defi

# Defining the parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('lamb', 0.000092, 'The regularization parameter')
tf.app.flags.DEFINE_integer('device_num', 8, 'Number of devices')
tf.app.flags.DEFINE_integer('max_device_num', 12, 'Number of devices')
tf.app.flags.DEFINE_integer('port', 8888, 'The begin port')

# Giving the parameters
device_num = FLAGS.device_num
max_device_num = FLAGS.max_device_num
port = FLAGS.port
lamb = FLAGS.lamb
complexity_sample = 9000

# Loading the training samples
x_data = np.loadtxt(defi.x_data_store).astype(np.float32)[0:complexity_sample]
y_data = np.loadtxt(defi.y_data_store).astype(np.float32)[0:complexity_sample]

# If it is indivisible, we drop the decimal part
batch_on_computers = complexity_sample / max_device_num

# Assigning the computers used for computing
computers = ['localhost:%d' % i for i in range(port, port + device_num)]
cluster = tf.train.ClusterSpec({'local': computers})
server = tf.train.Server(cluster, job_name="local", task_index=0)

slice_data = []
for i in range(max_device_num):
    slice_data.append(np.array(x_data[i:i + batch_on_computers, :]))
add_slice_length = np.array(x_data[device_num * batch_on_computers:]).shape[0] / device_num
slice_label = []
for i in range(max_device_num):
    slice_label.append(np.array(y_data[i:i + batch_on_computers]))

# Initializing the varaibles
init = tf.global_variables_initializer()

data_used_in_kernel = []
ana_result_used_in_kernel = []
device_index = 0

# Computing on the distributed computers virtually
for device_index in range(device_num):
    device = '/job:local/task:%d' % (device_index + 1)
    print(
            "============================= The current computation on " + device + ", the results are as follows =============================")
    with tf.device(device):
        # Launching the graph
        with tf.Session() as sess:
            sess.run(init)

            # Choosing Data
            handle_data = np.append(np.array(slice_data[device_index]), x_data[
                                                                        device_num * batch_on_computers + device_index * add_slice_length:device_num * batch_on_computers + (
                                                                                    device_index + 1) * add_slice_length],
                                    axis=0)
            label_data = np.append(np.array(slice_label[device_index]), y_data[
                                                                        device_num * batch_on_computers + device_index * add_slice_length:device_num * batch_on_computers + (
                                                                                    device_index + 1) * add_slice_length],
                                   axis=0)

            # The first dimension of choosed data x, that is, the number of samples x
            row_num = handle_data.shape[0]

            # Here, handle_data is like a matrix with the dimension n times d. Here, n is the number of samples x
            Trans_X_dot_X = np.dot(handle_data, handle_data.transpose())

            # Here, handle_data is like a matrix with the dimension n times d. Here, n is the number of samples x
            Inv_On_Trans_X_dot_X_Identity = np.linalg.inv(
                np.add(np.power(Trans_X_dot_X, 3),
                       np.multiply(complexity_sample / device_num, np.multiply(lamb, np.identity(row_num)))))

            data_used_in_kernel.append(handle_data.transpose())
            ana_result_used_in_kernel.append(np.dot(Inv_On_Trans_X_dot_X_Identity, label_data))
            sess.close()
            del sess

with tf.Session(server.target) as sess:
    print("Computing the risk of average models on " + server.target)
    print("===================================== Empirical risks =====================================")
    predicting_data = 0
    for i in range(device_num):
        data_computing = np.power(np.dot(x_data, data_used_in_kernel[i]), 3)
        predicting_data = predicting_data + np.multiply(1.0 / device_num,
                                                        np.dot(data_computing, ana_result_used_in_kernel[i]))
    # print predicting_data
    emp = sum(np.square(predicting_data - y_data)) / defi.complexity_sample

    print("===================================== Test errors =====================================")

    # Loading the testing samples
    x_test_data = np.loadtxt(defi.x_test_data_store).astype(np.float32)
    y_test_data = np.loadtxt(defi.y_test_data_store).astype(np.float32)

    predicting_test_data = 0
    for i in range(device_num):
        data_computing = np.power(np.dot(x_test_data, data_used_in_kernel[i]), 3)
        predicting_test_data = predicting_test_data + np.multiply(1.0 / device_num,
                                                                  np.dot(data_computing, ana_result_used_in_kernel[i]))
    # print predicting_test_data
    emp_exp = np.sum(np.square(predicting_test_data - y_test_data)) / defi.complexity_sample_test

    fo = open("../Data/LambdaIncrease.txt", "a+")
    fo.write(str(lamb) + ",")
    fo.write(str(complexity_sample) + ",")
    fo.write(str(device_num) + ",")
    fo.write(str(emp) + ",")
    fo.write(str(emp_exp) + ",")
    fo.write(str(emp_exp - emp) + "\n")
    fo.close()
    print "The computation on " + server.target + " is finished!"
    sess.close()
    del sess
