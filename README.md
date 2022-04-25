## Stability-Based Generalization Analysis of Distributed Learning Algorithms for Big Data

> As one of the efficient approaches to deal with big data, divide-and-conquer distributed algorithms, such as the distributed kernel regression, bootstrap, structured perception training algorithms, and so on, are proposed and broadly used in learning systems. Some learning theories have been built to analyze the feasibility, approximation, and convergence bounds of these distributed learning algorithms. However, less work has been studied on the stability of these distributed learning algorithms. In this paper, we discuss the generalization bounds of distributed learning algorithms from the view of algorithmic stability. First, we introduce a definition of uniform distributed stability for distributed algorithms and study the distributed algorithms' generalization risk bounds. Then, we analyze the stability properties and generalization risk bounds of a kind of regularization-based distributed algorithms. Two generalization distributed risks obtained show that the generalization distributed risk bounds for the difference between their generalization distributed and empirical distributed/leave-one-computer-out risks are closely related to the size of samples n and the amount of working computers m as O(m/n 1/2 ) . Furthermore, the results in this paper indicate that, for a good generalization regularized distributed kernel algorithm, the regularization parameter λ should be adjusted with the change of the term m/n 1/2 . These theoretic discoveries provide the useful guidance when deploying the distributed algorithms on practical big data platforms. We explore our theoretic analyses through two simulation experiments. Finally, we discuss some problems about the sufficient amount of working computers, nonequivalence, and generalization for distributed learning. We show that the rules for the computation on one single computer may not always hold for distributed learning.
 
---

## How to Cite

Xinxing Wu, Junping Zhang, Wang Fei-Yue. Stability-based Generalization Analysis of Distributed Learning Algorithms for Big Data. IEEE transactions on neural networks and learning systems, 2019, 31 (3), 801-812.
􏰃Link: https://ieeexplore.ieee.org/document/8709753


---
## About Codes


The folders "Fig2a", "Fig2b","Fig3","Fig4a","Fig4b" and "Fig5" are about the simulations.

The folder “Samples” includes the

1) weights https://github.com/xinxingwu/Distributed-Simulation/tree/master/Samples/Weights

2) testing samples https://github.com/xinxingwu/Distributed-Simulation/tree/master/Samples/TestingSamples

3) training samples with noise (0.5) https://github.com/xinxingwu/Distributed-Simulation/tree/master/Samples/TrainingSamplesWithNoise0.5

4) training samples with noise (1) https://github.com/xinxingwu/Distributed-Simulation/tree/master/Samples/TrainingSamplesWithNoise1

In addition, if using PyCharm to run the codes, it needs to configure the Project Interpreter. Please see https://github.com/xinxingwu/Distributed-Simulation/blob/master/Supplyment/PycharmEnvironmentSetting.gif
