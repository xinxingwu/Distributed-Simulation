#coding:utf-8
#SimulationForEmpdis.py, this program is used to train the model. Furthermore, we also use this program to obtain the expected error, empirical error and so on
import tensorflow as tf
import numpy as np
import Defi.Definitions as defi

#Defining the parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('training_epochs', 80,'Steps to train and print loss')
tf.app.flags.DEFINE_integer('display_step', 20,'Steps to disply the process of training')
tf.app.flags.DEFINE_integer('device_num', 96,'Number of devices')
tf.app.flags.DEFINE_integer('w_column_length', 2,'The column length of weight')
tf.app.flags.DEFINE_integer('port', 8888,'The begin port')
tf.app.flags.DEFINE_float('lamb', 0.04,'The regularization parameter')

#Giving the parameters
learning_rate=FLAGS.learning_rate
training_epochs=FLAGS.training_epochs
display_step=FLAGS.display_step
device_num=FLAGS.device_num
w_column_length=FLAGS.w_column_length
complexity_sample=defi.complexity_sample
batch_on_computers=complexity_sample/device_num
port=FLAGS.port
lamb=FLAGS.lamb

#Assigning the computers used for computing
computers=['localhost:%d' %i for i in range(port,port+device_num)]
cluster=tf.train.ClusterSpec({'local':computers})
server=tf.train.Server(cluster,job_name="local",task_index=0)

#Generating the training data
#Generating automatically
#x_data =np.float32([np.random.rand(complexity_sample),np.random.randn(complexity_sample)])
x_data=np.loadtxt(defi.x_data_store).astype(np.float32)
#Generating automatically
#y_data =np.dot([0.2,-0.3],x_data)+2+np.random.normal(0, 0.05)
y_data=np.loadtxt(defi.y_data_store).astype(np.float32)

#The array used to store the weights and bias from the distributed computers
ave_w=np.zeros((device_num,w_column_length))
ave_b=np.zeros(device_num)

#The two variables are used to store the average value of weights and bias from the distributed computers, respectively
b_for_average_model=0;
w_for_average_model=0;

#The array used to store the weights and bias from the one single computers
total_w=np.zeros(w_column_length)
total_b=np.zeros(1)

#Tensorflow's graph input
X=tf.placeholder("float")
Y=tf.placeholder("float")

device_index=0
#Giving the model
#Generating automatically
#W = tf.Variable(tf.random_uniform([1,w_column_length], -1.0, 1.0))
W =tf.Variable((tf.ones([1,w_column_length])*0.5))
b = tf.Variable(tf.zeros([1]))
y = tf.add(tf.matmul(W,X),b)

#Defining the Loss function
loss = tf.reduce_mean(tf.square(y - Y))
loss = loss+lamb*tf.norm([W])*tf.norm([W])
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#Initializing the varaibles
init = tf.global_variables_initializer()

#Training model on the distributed computers virtually
for device_index in range(device_num):
    device = '/job:local/task:%d' % device_index
    print("============================= The current computation on "+device+", the results are as follows =============================")
    with tf.device(device):
        # Launching the graph
        with tf.Session() as sess:
            sess.run(init)
            # Fitting all training data
            x_data_fetch = x_data[:,batch_on_computers*device_index:batch_on_computers*(device_index+1)]
            y_data_fetch = y_data[batch_on_computers*device_index:batch_on_computers*(device_index+1)]
            for epoch in range(training_epochs):
                sess.run(optimizer, feed_dict={X: x_data_fetch, Y: y_data_fetch})
                # Displaying logs per epoch step
                if (epoch + 1) % display_step == 0:
                    c = sess.run(loss, feed_dict={X: x_data_fetch, Y: y_data_fetch})
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W", sess.run(W), "b=", sess.run(b))
            print("============================= Optimization Finished! =============================\n")
            for i in range(w_column_length):
                ave_w[device_index][i]=sess.run(W)[0][i]
            ave_b[device_index]=sess.run(b)
            #sess.close()
            #del sess

#Training model on the one single computer
with tf.device('/job:local/task:0'):
    print("============================= The current total computation on the single /job:local/task:0, the results are as follows =============================")
    with tf.Session() as sess:
        sess.run(init)
        # Fitting all training data
        for epoch in range(training_epochs):
            sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
            # Displaying logs per epoch step
            if (epoch + 1) % display_step == 0:
                c = sess.run(loss, feed_dict={X: x_data, Y: y_data})
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W", sess.run(W), "b=", sess.run(b))
        print("============================= Optimization Finished! =============================\n")
        for i in range(w_column_length):
            total_w[i]=sess.run(W)[0][i]
        total_b[0]=sess.run(b)
        #sess.close()
        #del sess

#Computing the empirical risk, expected risk and the difference between the expected risk and the empirical risk
with tf.Session(server.target) as sess:
    print("Computing the risk of average models on "+server.target)
    w_for_average_model=np.mean(ave_w, axis=0)
    b_for_average_model=np.mean(ave_b, axis=0)
    emp=np.sum(np.square(np.matmul(w_for_average_model, x_data) + b_for_average_model-y_data))/complexity_sample
    emp_exp=(w_for_average_model[0]-2)*(w_for_average_model[0]-2)*(4.0/3)+2*(w_for_average_model[0]-2)*(b_for_average_model-2)+(w_for_average_model[1]+1)*(w_for_average_model[1]+1)+(b_for_average_model-2)*(b_for_average_model-2)+0.1
    print("The empirical risk is "+str(emp))
    print("The expected risk is " + str(emp_exp))
    print ("The difference between the expected risk and the empirical risk is "+str(emp_exp-emp))
    fo = open("../Data/resultforEmpdis.txt", "a+")
    fo.write(str(device_num) + ",")
    fo.write(str(emp) + ",")
    fo.write(str(emp_exp) + ",")
    fo.write(str(emp_exp - emp)+"\n")
    fo.close()
    print("The computation on "+server.target+" is finished!")
    #sess.close()
    #del sess