#coding:utf-8
#ShowGraphForCompatibilityWithAlpha.py, this program is used to show the curves of kinds of Compatibility graphs with Alpha
import numpy as np
import matplotlib.pyplot as plt

#The font size on the graph we will plot
size_font=18
#The marker size on the graph we will plot
size_marker=9
#Opening a txt file

f1 = open("../Data/eqv1.txt")
f2 = open("../Data/eqv2.txt")
alpha_list=[]
sum1_lists=[]
sum2_lists=[]
line1 = f1.readline()
line2 = f2.readline()

while line1:
    line1=line1.strip().split(",")
    line2 = line2.strip().split(",")
    alpha_list.append(line1[0])
    sum1_lists.append(line1[1])
    sum2_lists.append(line2[1])
    line1 = f1.readline()
    line2 = f2.readline()

alpha_list_data=np.array(alpha_list)
alpha_list_data=alpha_list_data.astype(float)
print(alpha_list_data)


sum1_lists_data=np.array(sum1_lists)
sum1_lists_data=sum1_lists_data.astype(float)
sum2_lists_data=np.array(sum2_lists)
sum2_lists_data=sum2_lists_data.astype(float)
print(sum1_lists_data+sum2_lists_data)

f1.close()
f2.close()

#Graphic display
plt.figure(figsize=(12,7))
plt.plot(alpha_list_data[0:11], sum1_lists_data[0:11]+sum2_lists_data[0:11], 'kx', linestyle='-',label='10 vs. 9',markersize=size_marker)
plt.plot(alpha_list_data[11:22], sum1_lists_data[11:22]+sum2_lists_data[11:22], 'bo', linestyle='--',label='20 vs. 19',markersize=size_marker)
plt.plot(alpha_list_data[22:33], sum1_lists_data[22:33]+sum2_lists_data[22:33], 'rs', linestyle=':', label='30 vs. 29',markersize=size_marker)

#plots the x lable
plt.xticks(fontsize=size_font)
plt.xlabel(r'$\alpha$',fontsize=size_font)
# plots the y lable
plt.yticks(fontsize=size_font)
plt.ylabel("$\mathrm{I}_{reg}^{\Delta_{S,z_{i}}}+\mathrm{I}_{reg}^{S}$",fontsize=size_font)
plt.legend(fontsize=size_font, loc = 'upper left')
#plt.savefig('figure.eps')
plt.show()
