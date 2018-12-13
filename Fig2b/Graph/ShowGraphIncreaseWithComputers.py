# coding:utf-8
# This program is used to show the curves of Figure 3 in the paper
import numpy as np
import matplotlib.pyplot as plt

# The font size on the graph we will plot
size_font = 18

# The marker size on the graph we will plot
size_marker = 9

# Opening a txt file
f = open("../Data/ComputerIncrease.txt")

computer_num_lists = []
emp_lists = []
emp_test_lists = []
emp_test_minus_emp_lists = []
line = f.readline()

while line:
    line = line.strip().split(",")
    computer_num_lists.append(line[1])
    emp_lists.append(line[2])
    emp_test_lists.append(line[3])
    emp_test_minus_emp_lists.append(line[4])
    line = f.readline()

computer_num_data = np.array(computer_num_lists)
computer_num_data = computer_num_data.astype(float)
print computer_num_data

emp_data = np.array(emp_lists)
emp_data = emp_data.astype(float)
print emp_data

emp_exp_data = np.array(emp_test_lists)
emp_exp_data = emp_exp_data.astype(float)
print emp_exp_data

emp_exp_minus_emp_data = np.array(emp_test_minus_emp_lists)
emp_exp_minus_emp_data = emp_exp_minus_emp_data.astype(float)
print emp_exp_minus_emp_data

f.close()

# Graphic display
plt.figure(figsize=(12, 7))
plt.plot(computer_num_data[0:10], emp_exp_data[0:10], 'bo', linestyle='--',
         label='Test error', markersize=size_marker)
plt.plot(computer_num_data[0:10], emp_exp_minus_emp_data[0:10], 'rs', linestyle=':',
         label='Difference between test error and empirical risk', markersize=size_marker)

show_value = str(0.05222)
plt.annotate(show_value, xytext=(2.02, 0.0562), xy=(2, 0.052220303852689234))
plt.plot(2, 0.052220303852689234,'ys')

# Ploting the x lable
plt.xticks(fontsize=size_font)
plt.xlabel("Amount of working computers", fontsize=size_font)

# Ploting the y lable
plt.yticks(fontsize=size_font)
plt.ylabel("Risk / Error", fontsize=size_font)
plt.legend(fontsize=size_font, loc='upper left')
plt.show()
