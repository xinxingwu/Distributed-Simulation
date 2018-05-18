#coding:utf-8
#ShowGraphForRiskWithLamb.py, using this code to show the kinds of risk graphs with the number of computers
import numpy as np
import matplotlib.pyplot as plt

size_font=18
size_marker=9
f = open("../Data/resultforEmpdis.txt")
lamb_num_lists=[]
emp_lists=[]
emp_exp_lists=[]
emp_exp_minus_emp_lists=[]
line = f.readline()

while line:
    line=line.strip().split(",")
    lamb_num_lists.append(line[0])
    emp_lists.append(line[1])
    emp_exp_lists.append(line[2])
    emp_exp_minus_emp_lists.append(line[3])
    line = f.readline()

lamb_num_data=np.array(lamb_num_lists)
lamb_num_data=lamb_num_data.astype(float)
print lamb_num_data

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
#plt.plot(lamb_num_data[22:30], emp_data[22:30], 'yo', linestyle='-',label='Empirical risk (4)')
#plt.plot(lamb_num_data[22:30], emp_exp_data[22:30], 'go', linestyle='-',label='Expected risk (4)')
#plt.plot(lamb_num_data[0:14], emp_data[0:14], 'ro', linestyle='-',label='Empirical risk (32)')
#plt.plot(lamb_num_data[0:14], emp_exp_data[0:14], 'go', linestyle='-',label='Expected risk (32)')
#plt.plot(lamb_num_data[8:15], emp_exp_minus_emp_data[8:15], 'bs', linestyle='-', label='The difference between expected and empirical risk (32)')
#plt.plot(lamb_num_data[30:36], emp_exp_minus_emp_data[30:36], 'r^', linestyle='-', label='The difference between expected and empirical risk (6)')
plt.plot(lamb_num_data, emp_exp_minus_emp_data, 'rs', linestyle=':', label='Difference between expected and empirical risk',markersize=size_marker)
# plots an x lable
plt.xticks(fontsize=size_font)
plt.xlabel("Regularization parameter",fontsize=size_font)
# plots an y lable
plt.yticks(fontsize=size_font)
plt.ylabel("Risk",fontsize=size_font)
plt.legend(fontsize=size_font, loc = 'upper left')
plt.show()