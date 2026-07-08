I am exploring the potential of quantum machine learning for meteo based on this paper https://arxiv.org/abs/2404.08737    
The code was not available, so I decided to do it by myself and open source it  
The creation of the dataset is done inside quantum_bve_step_by_step.ipynb  
The main algorithm of physical information neural network from experiment1 you must focus on is running_exp1 (4).ipynb      
Here is the sum up of the paper made by myself ![Paper summary](https://raw.githubusercontent.com/CyrilDeloince/qml_PINN_pasqal/main/images/1778833060073.jpg)        
Here is the results of experiment 1 that match exactly the results of the paper.   
You'll need to go through quantum_bve_step_by_step.ipynb and running_exp1 (4).ipynb to understand how to code it  
<img width="1681" height="637" alt="image" src="https://github.com/user-attachments/assets/d669afb4-fe9a-4396-bbd5-077c4dedb55f" />  

Let's me drive you now on the research work : how can we implement this photonically
In the original paper/Qadence implementation, the model is a qubit-based QNN. The circuit is built from: 
qubits + Ry feature map + HEA ansatz + sum-Z observable 

First modification : sum-Z observable 

The final prediction is obtained by measuring the total magnetisation observable:  
C = sum_m Z_m  
where Z_m is the Pauli-Z observable acting on qubit m. 

In our MerLin photonic version, we translate this readout into a dual-rail photonic representation. Each logical qubit is encoded using one photon distributed over two optical modes: 
|0>_L = |1,0>     
|1>_L = |0,1>  

In this encoding, the Pauli-Z measurement has a direct photon-number equivalent: 
Z_m  <=>  n_left_m - n_right_m             
This is equivalent because the left rail represents logical |0>, which has Pauli-Z eigenvalue +1, while the right rail represents logical |1>, which has Pauli-Z eigenvalue -1: 
|1,0> -> n_left - n_right = 1 - 0 = +1     
|0,1> -> n_left - n_right = 0 - 1 = -1      

Therefore, the qubit observable used in the paper, 
sum_m Z_m             

is implemented photonicly as:        
sum_m (<n_left_m> - <n_right_m>)          

This preserves the same measured quantity as the original QNN, but expressed in the natural measurement basis of a dual-rail photonic circuit. Thus, the readout remains faithful to main.tex: it is fixed, Hermitian, real-valued, and directly equivalent to the original total magnetisation observable. 
