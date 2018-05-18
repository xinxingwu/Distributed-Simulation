#coding:utf-8
#ShowGraphForRiskWithDeviceNum.py, this program is used to show the curves of kinds of error graphs with the number of computers
import numpy as np
import matplotlib.pyplot as plt

#The font size on the graph we will plot
size_font=18
#The marker size on the graph we will plot
size_marker=9
#Opening a txt file
f = open("../Data/resultforEmpdis.txt")
device_num_lists=[]
emp_lists=[]
emp_exp_lists=[]
emp_exp_minus_emp_lists=[]
line = f.readline()

while line:
    line=line.strip().split(",")
    device_num_lists.append(line[0])
    emp_lists.append(line[1])
    emp_exp_lists.append(line[2])
    emp_exp_minus_emp_lists.append(line[3])
    line = f.readline()

device_num_data=np.array(device_num_lists)
device_num_data=device_num_data.astype(int)
print(device_num_data)

emp_data=np.array(emp_lists)
emp_data=emp_data.astype(float)
print(emp_data)

emp_exp_data=np.array(emp_exp_lists)
emp_exp_data=emp_exp_data.astype(float)
print(emp_exp_data)

emp_exp_minus_emp_data=np.array(emp_exp_minus_emp_lists)
emp_exp_minus_emp_data=emp_exp_minus_emp_data.astype(float)
print(emp_exp_minus_emp_data)

f.close()

#Graphic display
plt.figure(figsize=(12,7))
plt.plot(device_num_data, emp_data, 'kx', linestyle='-',label='Empirical risk',markersize=size_marker)
plt.plot(device_num_data, emp_exp_data, 'bo', linestyle='--',label='Expected risk',markersize=size_marker)
plt.plot(device_num_data, emp_exp_minus_emp_data, 'rs', linestyle=':', label='Difference between expected and empirical risk',markersize=size_marker)

#The range of y
plt.ylim((0, 1.2))
#plots the x lable
plt.xticks(fontsize=size_font)
plt.xlabel("Number of computers",fontsize=size_font)
# plots the y lable
plt.yticks(fontsize=size_font)
plt.ylabel("Risk",fontsize=size_font)
plt.legend(fontsize=size_font, loc = 'upper left')
#plt.savefig('figure.eps')
plt.show()