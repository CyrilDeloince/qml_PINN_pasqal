I am exploring the potential of quantum machine learning for weather forecasting, based on the paper [arXiv:2404.08737](https://arxiv.org/abs/2404.08737).

The original code was not publicly available, so I reimplemented the pipeline myself and decided to open-source it.

The dataset generation is implemented in [`quantum_bve_step_by_step.ipynb`](quantum_bve_step_by_step.ipynb).  
The main Experiment 1 QNN/PINN-style training notebook is [`running_exp1 (4).ipynb`](running_exp1%20(4).ipynb).

Here is my summary of the paper:

![Paper summary](https://raw.githubusercontent.com/CyrilDeloince/qml_PINN_pasqal/main/images/1778833060073.jpg)

Here are the Experiment 1 results, which reproduce the results reported in the paper:

<img width="1681" height="637" alt="Experiment 1 results" src="https://github.com/user-attachments/assets/d669afb4-fe9a-4396-bbd5-077c4dedb55f" />

## Photonic adaptation

The next research step is to investigate how the original qubit-based QNN can be implemented in a photonic setting.

In the paper, the model is a qubit-based QNN built from:

- qubits,
- an $R_y$ feature map,
- a hardware-efficient ansatz,
- a total magnetisation observable, $\sum_m Z_m$.

### First modification: sum-Z observable

The final prediction in the original model is obtained by measuring the total magnetisation observable:

$$
C = \sum_m Z_m,
$$

where $Z_m$ is the Pauli-Z observable acting on qubit $m$.

In the MerLin photonic version, we translate this readout into a dual-rail photonic representation. Each logical qubit is encoded using one photon distributed over two optical modes:

$$
|0\rangle = |1,0\rangle,
$$

$$
|1\rangle = |0,1\rangle.
$$

In this encoding, the Pauli-Z measurement has a direct photon-number equivalent:

$$
Z_m \equiv n_{\mathrm{left},m} - n_{\mathrm{right},m}.
$$

This is equivalent because the left rail represents logical $|0\rangle$, which has Pauli-Z eigenvalue $+1$:

$$
|1,0\rangle \rightarrow n_{\mathrm{left}} - n_{\mathrm{right}} = 1 - 0 = +1,
$$

while the right rail represents logical $|1\rangle$, which has Pauli-Z eigenvalue $-1$:

$$
|0,1\rangle \rightarrow n_{\mathrm{left}} - n_{\mathrm{right}} = 0 - 1 = -1.
$$

Therefore, the qubit observable used in the paper,

$$
\sum_m Z_m,
$$

is implemented photonicly as

$$
\sum_m \left(\langle n_{\mathrm{left},m} \rangle - \langle n_{\mathrm{right},m} \rangle\right).
$$

This preserves the same measured quantity as the original QNN, but expresses it in the natural measurement basis of a dual-rail photonic circuit.

The readout therefore remains faithful to the original paper: it is fixed, Hermitian, real-valued, and directly equivalent to the original total magnetisation observable.

Here, $a$ and $b$ are trainable classical parameters initialized from the mean and standard deviation of the target data. This matches the paper's output scaling while using the photonic observable defined above.



