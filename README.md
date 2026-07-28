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


### Second modification: output scaling

The original paper first defines the raw QNN output as the expectation value of the measured observable:

$$
\tilde{y}_i =
\langle \Phi(\vec{x}_i) | \hat{C} | \Phi(\vec{x}_i) \rangle .
$$

For the weather task, this output is used to predict the stream function \(\psi\).  
However, the paper also states that the measured QNN output is passed through a learnable affine transformation:

$$
\psi =
\alpha_{n-1}\mathrm{QNN}(\lambda,\phi,t) + \alpha_n .
$$

In the photonic version, after replacing the qubit observable \(\sum_m Z_m\) by its dual-rail photonic equivalent,

$$
\sum_m \left(n_{\mathrm{left},m} - n_{\mathrm{right},m}\right),
$$

we keep the same output-scaling structure:

$$
\hat{\psi} =
\alpha_{\mathrm{scale}}
\langle C_{\mathrm{photonic}} \rangle
+
\alpha_{\mathrm{shift}} .
$$

The two parameters alpha_scale and alpha_shift are trainable classical parameters, initialized from the standard deviation and mean of the target stream-function data.

This is therefore not an additional photonic assumption: it is the same learnable output transformation used in the original paper, applied after the photonic equivalent of the original sum_m Z_m observable.



### Third modification: Trainable Photonic Entanglement

In the standard qubit architecture, inter-qubit entanglement is achieved via fixed **CNOT** gates at zero cost in trainable parameters. A CNOT operation acts on two logical qubits in dual-rail encoding (4 optical modes) as a conditional swap:

$$\text{CNOT} |\psi_{\text{control}}, \psi_{\text{target}}\rangle \quad \text{flips target iff } \text{control} = |1\rangle$$

This operation is **non-linear in photon number**: it requires direct photon-photon interactions, which are physically impossible in passive linear optics without non-linear media or ancillary photons with measurement feed-forward (*KLM theorem, 2001*).

In our dual-rail photonic circuit implementation, inter-qubit coupling is performed using a beam splitter (BS) between one mode of qubit $q$ and one mode of qubit $q+1$:

$$\text{BS}(\theta): \text{mixes mode } (2q+1) \text{ with mode } 2(q+1)$$

#### The limitation of fixed beam splitters ($\theta = \pi/4$)
When $\theta$ is fixed at $\pi/4$, it provides only partial, passive mixing: a single photon reaching the beam splitter has a 50% probability of remaining in its original mode and a 50% probability of transitioning to the adjacent qubit's mode. This setup is fundamentally weaker than a CNOT gate due to two major drawbacks:
1. **Subspace Leakage:** It can cause a photon to "leak" outside the valid single-photon-per-qubit dual-rail subspace (e.g., qubit $q$ loses its photon while qubit $q+1$ absorbs two photons).
2. **Unconditional Interaction:** The state transformation occurs unconditionally, regardless of the control qubit's state.

#### The Solution: trainable photonic entanglement
To compensate for these physical constraints, we convert these passive beam splitters into **trainable entangling blocks**: 

$$\text{BS}(\theta_{\text{trainable}}, \phi_{\text{trainable}})$$

This parameterization enables the optimization algorithm to learn optimal coupling strengths, maximizing effective entanglement within the valid dual-rail subspace while actively suppressing population leakage into unphysical optical states. Therefore, a few additional parameters are required to reach equivalent expressivity.
