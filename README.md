Stabilizer Tensor Networks: Universal Quantum Simulator on a Basis of Stabilizer States

Public release of code, v1.2

The main class to use is "gen_clifford", a generalization inheriting Qiskit's "Clifford" class. This and other supporting functions can be found in stabilizers.py. 

Its general use follows qiskit's Clifford class (see https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Clifford), initializing via Circuits with only Clifford gates or Clifford tableaus, among others.
The modified "compose" method accepts any unitary even if it's non-Clifford, which is decomposed with our own methods 
It stores the vector of complex coefficients for a quantum state in MPS form, using quimb (see https://quimb.readthedocs.io/en/latest/tensor-1d.html), while using the tableau in the inherited Clifford class to store the Clifford gates. They can be understood as a change of basis for the MPS.

We are able to disentangle further the MPS with a generalization of the exact method in arxiv:2412.17209 and an implementation of the sweeping disentangling in arxiv:2407.01692.

The stabilizers.ipynb file is an example in notebook format on its utilization, as well as the code lines to produce the figure in the main text of https://arxiv.org/abs/2403.08724, which illustrates how to do use the class methods to achieve the procedures above.
