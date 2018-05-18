#coding:utf-8
#ShowGraphForRiskWithN.py, using this code to show the kinds of risk graphs with the number of computers
import numpy as np
import matplotlib.pyplot as plt

size_font=18
size_marker=9
f = open("../Data/resultforEmpdisOnN.txt")
smaple_num_lists=[]
emp_lists=[]
emp_exp_lists=[]
emp_exp_minus_emp_lists=[]
line = f.readline()

while line:
    line=line.strip().split(",")
    smaple_num_lists.append(line[0])
    emp_lists.append(line[1])
    emp_exp_lists.append(line[2])
    emp_exp_minus_emp_lists.append(line[3])
    line = f.readline()

sample_num_data=np.array(smaple_num_lists)
sample_num_data=sample_num_data.astype(int)
print sample_num_data

emp_data=np.array(emp_lists)
emp_data=emp_data.astype(float)
print emp_data

emp_exp_data=np.array(emp_exp_lists)
emp_exp_data=emp_exp_data.astype(float)
print emp_exp_data

emp_exp_minus_emp_data=np.array(emp_exp_minus_emp_lists)
emp_exp_minus_emp_data=emp_exp_minus_emp_data.astype(float)
print emp_exp_minus_emp_data

f.close()

# Graphic display
plt.figure(figsize=(12,7))
plt.plot(sample_num_data, emp_data, 'kx', linestyle='-',label='Empirical risk',markersize=size_marker)
plt.plot(sample_num_data, emp_exp_data, 'bo', linestyle='--',label='Expected risk',markersize=size_marker)
plt.plot(sample_num_data, emp_exp_minus_emp_data, 'rs', linestyle=':', label='Difference between expected and empirical risk',markersize=size_marker)
# plots an x lable
plt.ylim((0.23, 0.68))
plt.xticks(fontsize=size_font)
plt.xlabel("Size of sample",fontsize=size_font)
# plots an y lable
plt.yticks(fontsize=size_font)
plt.ylabel("Risk",fontsize=size_font)
plt.legend(fontsize=size_font, loc = 'upper left')
plt.show()