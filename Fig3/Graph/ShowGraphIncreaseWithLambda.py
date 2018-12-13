# coding:utf-8
# This program is used to show the curves of Figure 5 in the paper
import numpy as np
import matplotlib.pyplot as plt

# The font size on the graph we will plot
size_font = 18

# The marker size on the graph we will plot
size_marker = 9

# Opening a txt file
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
plt.plot(lambda_num_lists[0:22], emp_test_minus_emp_data[0:22], 'rs', linestyle='--',
         label='Difference between test error and empirical risk (6,10000)', markersize=size_marker)
plt.plot(lambda_num_lists[22:44], emp_test_minus_emp_data[22:44], 'bo', linestyle=':',
         label='Difference between test error and empirical risk (4,8000)', markersize=size_marker)

# Adding points
#show_value1 = str(0.06988)
#plt.annotate(show_value1, xytext=(7.55e-05, 0.07018), xy=(7.8e-05, 0.06988924985921091))

#show_value2 = str(0.06996)
#plt.annotate(show_value2, xytext=(8.4e-05, 0.07018), xy=(8.2e-05, 0.06996912216210177))

#show_value3 = str(0.06847)
#plt.annotate(show_value3, xytext=(7.55e-05, 0.0687), xy=(7.8e-05, 0.06847719100891894))

#show_value4 = str(0.06849)
#plt.annotate(show_value4, xytext=(8.4e-05, 0.0687), xy=(8.2e-05, 0.06849026576037105))

show_value5 = str(0.06853)
plt.annotate(show_value5, xytext=(0.0000956, 0.06875), xy=(0.0001, 0.068534939767174))
plt.plot(0.0001, 0.068534939767174,'ys')

show_value6 = str(0.07013)
plt.annotate(show_value6, xytext=(0.0000956, 0.07035), xy=(0.0001, 0.07013542410678936))
plt.plot(0.0001, 0.07013542410678936,'ks')

# Ploting the x lable
plt.xticks(fontsize=size_font)
plt.xlabel("Regularization parameter", fontsize=size_font)

# Ploting the y lable
plt.xlim(0.000013, 0.000102)
plt.yticks(fontsize=size_font)
plt.ylabel("Risk / Error", fontsize=size_font)
plt.legend(fontsize=size_font, loc='upper right')
plt.show()
