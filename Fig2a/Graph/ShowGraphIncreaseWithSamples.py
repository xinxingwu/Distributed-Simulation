# coding:utf-8
# This program is used to show the curves of Figure 2 in the paper
import numpy as np
import matplotlib.pyplot as plt

# The font size on the graph we will plot
size_font = 18

# The marker size on the graph we will plot
size_marker = 9

# Opening a txt file
f = open("../Data/SampleIncrease.txt")

sample_num_lists = []
emp_lists = []
emp_test_lists = []
emp_test_minus_emp_lists = []
line = f.readline()

while line:
    line = line.strip().split(",")
    sample_num_lists.append(line[0])
    emp_lists.append(line[1])
    emp_test_lists.append(line[2])
    emp_test_minus_emp_lists.append(line[3])
    line = f.readline()

sample_num_data = np.array(sample_num_lists)
sample_num_data = sample_num_data.astype(float)
print sample_num_data

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
plt.plot(sample_num_data[0:14], emp_test_data[0:14], 'bo', linestyle='--',
         label='Test error', markersize=size_marker)
plt.plot(sample_num_data[0:14], emp_test_minus_emp_data[0:14], 'rs', linestyle=':',
         label='Difference between test error and empirical risk', markersize=size_marker)

show_value1 = str(0.06853)
plt.annotate(show_value1, xytext=(7500, 0.074), xy=(8000, 0.068534939767174))
plt.plot(8000, 0.068534939767174,'ys')


show_value2 = str(0.05762)
plt.annotate(show_value2, xytext=(9500, 0.063), xy=(10000, 0.057624356515303045))
plt.plot(10000, 0.057624356515303045,'ys')

# Ploting the x lable
plt.xticks(fontsize=size_font)
plt.xlabel("Size of samples", fontsize=size_font)

# Ploting the y lable
plt.yticks(fontsize=size_font)
plt.ylabel("Risk / Error", fontsize=size_font)
plt.legend(fontsize=size_font, loc='upper right')
plt.show()
