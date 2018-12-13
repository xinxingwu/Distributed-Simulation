#coding:utf-8
# This program is used to show the curves of Figure 4a in the paper
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
plt.plot(lambda_num_lists[20:40], emp_test_minus_emp_data[20:40], 'bo', linestyle=':',
         label='Difference (8,10000)', markersize=size_marker)

plt.plot(lambda_num_lists[40:60], emp_test_minus_emp_data[40:60], 'rs', linestyle=':',
         label='Difference (8,9000)', markersize=size_marker)


# Ploting the x lable
plt.xticks(fontsize=size_font)
plt.xlabel("Regularization parameter", fontsize=size_font)

# Ploting the y lable
plt.yticks(fontsize=size_font)
plt.ylabel("Risk / Error", fontsize=size_font)
plt.legend(fontsize=size_font, loc='upper right')
plt.show()
