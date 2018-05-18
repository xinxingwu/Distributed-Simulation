#coding:utf-8
#GenerationTrainingData.py, using this code to generate the train data
import numpy as np
import Defi.Definitions as defi

x_data =np.float32([np.random.rand(defi.complexity_sample),np.random.randn(defi.complexity_sample)])
np.savetxt(defi.x_data_store, x_data)
y_data =np.dot([2,-1],x_data)+2+np.random.normal(0, 0.1)
np.savetxt(defi.y_data_store, y_data)