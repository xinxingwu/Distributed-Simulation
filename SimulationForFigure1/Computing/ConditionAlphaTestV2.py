import numpy as np
import numpy.linalg as la
import Defi.Definitions as defi
import matplotlib.pyplot as plt

complexity_sample=defi.complexity_sample
lamb=0.04

x_data=np.loadtxt(defi.x_data_store).astype(np.float32)
y_data=np.loadtxt(defi.y_data_store).astype(np.float32)

#The font size on the graph we will plot
size_font=18
#The marker size on the graph we will plot
size_marker=9
#Opening a txt file
f = open("../Data/resultforEmpdis.txt")
device_num_lists=[]
w=[]
w_1=[]
w_2=[]
b=[]
line = f.readline()

while line:
    line=line.strip().split(",")
    device_num_lists.append(line[0])
    w_1.append(line[1])
    w_2.append(line[2])
    b.append(line[3])
    line = f.readline()

device_num_data = np.array(device_num_lists)
device_num_data = device_num_data.astype(int)
print(device_num_data)

w_1_data = np.array(w_1)
w_1_data = w_1_data.astype(float)
print(w_1_data)

w_2_data = np.array(w_2)
w_2_data = w_2_data.astype(float)
print(w_2_data)

b_data = np.array(b)
b_data = b_data.astype(float)
print(b_data)

w_for_average_1=np.array([w_1_data[0],w_2_data[0]])
print w_for_average_1
w_for_average_2=np.array([w_1_data[1],w_2_data[1]])
print w_for_average_2
w_for_average_11=np.array([w_1_data[0],w_2_data[0],b_data[0]])
w_for_average_21=np.array([w_1_data[1],w_2_data[1],b_data[1]])


alpha=0
while (alpha <=1):
    reg_1 = np.sum(np.square(np.matmul(w_for_average_1, x_data) + b_data[0] - y_data)) / complexity_sample + np.multiply(lamb, np.multiply(la.norm(w_for_average_11), la.norm(w_for_average_11)))
    print reg_1
    reg_1_plus=np.sum(np.square(np.matmul((1-alpha)*w_for_average_1+alpha*w_for_average_2, x_data) + (1-alpha)*b_data[0]+alpha*b_data[1] - y_data)) / complexity_sample + np.multiply(lamb, np.multiply(la.norm((1-alpha)*w_for_average_11+alpha*w_for_average_21), la.norm((1-alpha)*w_for_average_11+alpha*w_for_average_21)))
    print reg_1_plus

    fo = open("../Data/eqv2.txt", "a+")
    fo.write(str(alpha) + ",")
    fo.write(str(reg_1-reg_1_plus)+"\n")
    fo.close()
    alpha=alpha+0.1
