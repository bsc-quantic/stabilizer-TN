from stabilizers import *
from matplotlib import pyplot as plt
from time import time
import numpy as np
from qiskit.quantum_info import random_clifford
from random import random

import pickle
import sys

def max_inner_ind(tn):
    max_size = 0
    inds = tn.inner_inds()
    for ind in inds:
        new_size = tn.ind_size(ind)
        if new_size >= max_size: 
            max_size = new_size
    return max_size 

# Big non-stabilizer test
n_qubits = int(sys.argv[1])
total_depth = 2*n_qubits
tries = 1
mode = 'tn'
max_bond = None # this should work for not limiting max_bond


average = [0,]*total_depth
samples = []
average2 = [0,]*total_depth
samples2 = []
start = time()

for j in range(tries):
    bonds_dis = [1]
    bonds_dis2 = [1]
    new_cliff = random_clifford(n_qubits)
    stn_dis2 = gen_clifford(new_cliff,mode=mode,max_bond=max_bond,disentangle='exact+heuristic')
    stn_dis = gen_clifford(new_cliff,mode=mode,max_bond=max_bond,disentangle='exact+heuristic3')
    qc = QuantumCircuit(n_qubits)
    qc.t(int(n_qubits*random()))

    for depth in range(1,total_depth):
        print(f"Calculating depth {depth}")
                    
        try:
            stn_dis2.compose(qc)
            stn_dis.compose(qc)
        except:
            print(f"We had to rerun after t-gate {depth}")
            stn_dis2.compose(qc)
            stn_dis.compose(qc)
        bonds_dis2.append(max_inner_ind(stn_dis2.xvec))
        bonds_dis.append(max_inner_ind(stn_dis.xvec))

        new_cliff = random_clifford(n_qubits)
        stn_dis2.compose(new_cliff)
        stn_dis.compose(new_cliff)

    average2 = [average2[i] + bonds_dis2[i]/tries for i in range(total_depth)] 
    average = [average[i] + bonds_dis[i]/tries for i in range(total_depth)] 
    samples2.append(bonds_dis2)
    samples.append(bonds_dis)

print(time()-start)
time_stmp = time()
sep = str(time_stmp).find('.')
stmp = str(int(time_stmp))[sep-5:sep] + '_' + str(time_stmp)[sep+1:-1]

data = {'samples': samples, 'average': average}
data2 = {'samples': samples2, 'average': average2}
with open(f"data/stabilizer_dis_heur3_{stmp}.pickle", 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"data/stabilizer_dis_heur2_{stmp}.pickle", 'wb') as handle:
    pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)