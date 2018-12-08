#coding:utf-8
# This program is used to show the curves of Figure 6a in the paper
import numpy as np
import matplotlib.pyplot as plt

#The font size on the graph we will plot
size_font=18

#The marker size on the graph we will plot
size_marker=9

#Opening a txt file
f = open("../Data/LambdaIncrease.txt")

lambda_num_lists = []
emp_lists = []
emp_test_lists = []
emp_test_minus_emp_lists = []
line = f.readline()

while line:
    line = line.strip().split(",")
    lambda_num_lists.append(line[0])
    emp_lists.append(line[3])
    emp_test_lists.append(line[4])
    emp_test_minus_emp_lists.append(line[5])
    line = f.readline()

lambda_num_data = np.array(lambda_num_lists)
lambda_num_lists = lambda_num_data.astype(float)
print lambda_num_lists

emp_data = np.array(emp_lists)
emp_data = emp_data.astype(float)
print emp_data

emp_test_data = np.array(emp_test_lists)
emp_test_data = emp_test_data.astype(float)
print emp_test_data

emp_test_minus_emp_data = np.array(emp_test_minus_emp_lists)
emp_test_minus_emp_data = emp_test_minus_emp_data.astype(float)
print emp_test_minus_emp_data

f.close()

# Graphic display
plt.figure(figsize=(12, 7))
# Noise Sigma 0.5
plt.plot(lambda_num_lists[0:46:2], emp_test_minus_emp_data[0:46:2], 'rs', linestyle='--',
         label='Difference between test error and empirical risk ($\sigma^2=0.5$)', markersize=size_marker)
show_value1 = str(0.069758)
plt.annotate(show_value1,  xytext=(4.8e-05, 0.069858), xy=(5.4e-05, 0.069758250022623))
plt.plot(5.4e-05, 0.069758250022623,'ks')

# Noise Sigma 1
#plt.plot(lambda_num_lists[46:92:2], emp_test_minus_emp_data[46:92:2], 'bo', linestyle=':',
#         label='Difference between test error and empirical risk ($\sigma^2=1.0$)', markersize=size_marker)
#show_value2 = str(0.081935)
#plt.annotate(show_value2,  xytext=(0.000095, 0.08205), xy=(0.000102, 0.08193508007896423))
#plt.plot(0.000102, 0.08193508007896423,'ys')

# Ploting the y lable
plt.xticks(fontsize=size_font)
plt.xlabel("Regularization parameter", fontsize=size_font)

# Ploting the y lable
plt.yticks(fontsize=size_font)
plt.ylabel("Risk / Error", fontsize=size_font)
plt.legend(fontsize=size_font, loc='upper left')
plt.show()
