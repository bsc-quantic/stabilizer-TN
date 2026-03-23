from stabilizers import *
from matplotlib import pyplot as plt
from time import time
import numpy as np
from qiskit.quantum_info import random_clifford
from random import random

import sys
import pickle

def max_inner_ind(tn):
    max_size = 0
    inds = tn.inner_inds()
    for ind in inds:
        new_size = tn.ind_size(ind)
        if new_size >= max_size: 
            max_size = new_size
    return max_size

########### This argument sets the size N of the experiment ###########
N = int(sys.argv[1])
Ns = [N] # Ns = [6,8,10,12,14,16,18]
if N<50:
    t_gates = {N: 3*N + 1 for N in Ns}
else:
    t_gates = {N: 2*N + 5 for N in Ns}
t_gate_fid_points = {N: [int(0.1*N), int(0.5*N), int(N), int(1.5*N), int(2*N), int(3*N)] for N in Ns}
# part = str(sys.argv[2])

########### This argument sets the fraction of the angle to use (1 will be pi/8 angle, 0.01 will be 0.01*pi/8 angle, etc) ###########
param_str = str(sys.argv[2])
params = [np.float32(sys.argv[2])]

########### This argument is a string in ['p1','p2','p2a','p2b','p2c','p3','p3a','p3b','p3c'] that breaks down the simulation if it's too computationally expective to do in one go ###########
try:
    part = str(sys.argv[3])
except IndexError:
    part = ''

bond_dim = 128
n_bond_dim = int(np.log2(bond_dim))
samples = 20
if N>50:
    tries = 1
elif N==50 and params[0]==1.0 and param_str=='p3c':
    tries = 2
elif N>=32:
    tries = 3
elif N>16:
    tries = 6
else:
    tries = 10

bond_list_pre = list(set([int(d) for d in np.logspace(0,n_bond_dim,base=2,num=samples)]))
bond_list_pre.sort()
samples = len(bond_list_pre)
if part == 'p1':
    bond_list = bond_list_pre[:samples//2]
elif part == 'p2':
    bond_list = bond_list_pre[samples//2:-3]
elif part == 'p2a':
    bond_list = bond_list_pre[samples//2:samples//2+2]
elif part == 'p2b':
    bond_list = bond_list_pre[samples//2+2:samples//2+3]
elif part == 'p2c':
    bond_list = bond_list_pre[samples//2+3:samples//2+4]
elif part == 'p3':
    bond_list = bond_list_pre[-3:]
elif part == 'p3a':
    bond_list = bond_list_pre[-3:-2]
elif part == 'p3b':
    bond_list = bond_list_pre[-2:-1]
elif part == 'p3c':
    bond_list = bond_list_pre[-1:]
else:
    bond_list = bond_list_pre
    part = ''

print(f"Calculating for N={N}")

data_fid = {param: {t: [] for t in t_gate_fid_points[N]} for param in params}
data_fid_avg = {param: {t: [] for t in t_gate_fid_points[N]} for param in params}
data_local = {param: {t: [] for t in t_gate_fid_points[N]} for param in params}
data_bonds = {param: {t: [] for t in t_gate_fid_points[N]} for param in params}
data_entropy = {param: [] for param in params}
data_entropy_avg = {param: [] for param in params}
data_entropy_rescaled ={param:  [] for param in params}

N_start = time()

for param in params:
    angle = np.pi/8 * param
    print(f"Using angle {angle}")

    fidelities_list = data_fid[param]
    local_list = data_local[param]
    bonds_list = data_bonds[param]
    entropy = data_entropy[param]
    entropy_rescaled = data_entropy_rescaled[param]

    qc_rot  = QuantumCircuit(N)
    qc_rot.rz(2*angle, int(N*random()))

    qc_0 = QuantumCircuit(N)
    circuit_cliffs = []

    start = time()
    for t in range(t_gates[N]):
        circuit_cliffs.append(random_clifford(N))

    print('Skipping initial state')

    for j,bond in enumerate(bond_list):
        for t in range(tries):
            if bond==128 and N==100:
                t_gates[100] = int(1.6*N) + 2
            print(f"progress: bond {bond}")
            current_mps = gen_clifford(qc_0,max_bond=bond,contract=True,disentangle='exact+heuristic')
            for k in range(t_gates[N]):
                current_mps.compose(circuit_cliffs[k])
                try:
                    current_mps.compose(qc_rot,check_svd=True)
                except:
                    # print(f"We had to rerun bond {bond} after t-gate {k}")
                    current_mps.compose(qc_rot,check_svd=True)
                if k+1 in t_gate_fid_points[N]: 
                    if t==0:
                        bonds_list[k+1].append(bond)  # if m==0: bonds[i] = bond
                    trunc = 1
                    for tr in current_mps.truncations:
                        trunc *= tr
                    fidelities_list[k+1].append(trunc)
                    if t==0:
                        data_fid_avg[param][k+1].append(trunc/tries)
                    else:
                        data_fid_avg[param][k+1][j] += trunc/tries
                    if N<100:
                        local_list[k+1].append(current_mps.measure_obs((N//2-1)*'I' + 'X' + (N-N//2)*'I')[(N//2-1)*'I' + 'X' + (N-N//2)*'I']['ev'])
                if bond==bond_dim:
                    # print(f"storing entropy of maximum bond dim {bond}")
                    ent = max([current_mps.xvec.entropy(k) for k in range(1,N)])
                    entropy.append(ent)
                    entropy_rescaled.append(2*ent/N)
                    if t==0:
                        data_entropy_avg[param].append(ent/tries)
                    else:
                        data_entropy_avg[param][k] += ent/tries

    print(f"This parameter={angle} took {time()-start}")

    print(f"This N={N} took {time()-N_start}")

    data_array = [data_fid_avg,data_local,data_bonds,data_entropy_avg]

    time_stmp = time()
    sep = str(time_stmp).find('.')
    stmp = str(int(time_stmp))[sep-5:sep] + '_' + str(time_stmp)[sep+1:-1]

    with open(f"data/stabilizer_rot_trunc_{N}_{param_str}_{part}_{stmp}.pickle", 'wb') as handle:
        pickle.dump(data_array, handle, protocol=pickle.HIGHEST_PROTOCOL)