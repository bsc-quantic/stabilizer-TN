########################################## IMPORTS ###############################################

from qiskit import QiskitError, QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.quantum_info.operators.symplectic.clifford_circuits import *
from qiskit.circuit import Barrier, Delay, Gate, Instruction
from qiskit.circuit.exceptions import CircuitError
from qiskit.providers.fake_provider import FakeWashingtonV2 # GenericBackendV2(127)

import numpy as np
from random import random
from scipy.sparse import lil_array #, csr_array, coo_array
from time import time
import numbers

import quimb as qu
import quimb.tensor as qtn
from quimb import quimbify
from autoray import do
import autoray

################################## Needed gates from quimb #######################################

RX = qu.Rx
RZ = qu.Rz
CNOT = qu.controlled('not')
X = qu.pauli('X')
H = qu.hadamard()
Z = qu.pauli('Z')
S = qu.S_gate()
Sdg = qu.S_gate().conj()

##################################### Auxiliary functions ########################################

                            ### Glossary of concepts ###
# Boolean pauli form = [0,1,0,1 #qubits with an X ; 1,0,0,1 # qubits with a Z; 0 # phase] 
# Boolean clifford basis = entries of a clifford tableau (in boolean pauli form)
# gen_clifford class = extension of Qiskit's clifford class to non-clifford circuits


def connectivity_kyiv():
    # Uses fake chip Washington and adds two missing connections to get 
    # the connectivity of the IBM 127qb-chip experiment
    fake = FakeWashingtonV2() # GenericBackendV2(127)
    cx_instructions = []
    for instruction in fake.instructions:
        if instruction[0].name == 'cx':
            if instruction[1] not in cx_instructions and (instruction[1][1],instruction[1][0]) not in cx_instructions:
                cx_instructions.append(instruction[1])
    cx_instructions.append([109,114])
    cx_instructions.append([9,8])

    return cx_instructions




def multiply_bool_pauli(pauli1,pauli2):
    # Returns the multiplication of two Paulis in boolean X|Z form with the resulting phase at the end
    # !!! This does not fit the boolean form of the tableau because of the phase at the end !!!
    pauli = pauli1 # copy the first operator (only need the shape and the phase)
    total_qb = len(pauli)//2
    phase_mat = [[1,1,1,1],[1,1,1j,-1j],[1,-1j,1,1j],[1,1j,-1j,1]] # table of phases for each commutation (easy and fast)

    for i in range(len(pauli1)//2):
        pauli[-1] *= phase_mat[2*pauli1[i]+pauli1[i+total_qb]][2*pauli2[i]+pauli2[i+total_qb]]
        pauli[i],pauli[i+total_qb] = (pauli1[i] + pauli2[i])%2, (pauli1[i+total_qb] + pauli2[i+total_qb])%2
    pauli[-1] *= (-1)**(pauli2[-1]) # add the phase of the second pauli

    return pauli




def check_comm(vector,entry,complement,accum=None,qubits=None):
    # Checks if an operator (vector) in boolean clifford form commutes with another one (entry), usually extracted from a tableau.
    # Also stores or updates phase information (accum) that can be used to extract the phase of anticommuting entries.
    # This is needed when finding the decomposition of an operator in a given boolean clifford basis (tableau).
    comm = 1
    total_qb = len(vector)//2

    # In the general case we check the whole thing, but if we know (with method arguments)
    # which *qubits* to check we can save some time
    if qubits is None:
        qubits = range(total_qb)

    checks = [(qubit,(vector[qubit],vector[qubit+total_qb])) for qubit in qubits]
    for i,v in checks:
        comp = (int(entry[i]),int(entry[i+total_qb]))
        if v == (0,0) or comp == (0,0):
            continue
        if v != comp:
            comm *= -1

    # if comm is 1 then vector and given "entry" operator do not anti-commute
    if comm > 0:
        return 0, accum
    # otherwise, they do anti-commute and so "entry" operator is in the decomposition of vector
    else:
        if (accum is not None):
            accum = multiply_bool_pauli(accum,complement)
        else:
            accum = entry[:-1] + [(-1)**entry[1]]

        return 1, accum
    



def expect_tn(bra,G,ket,where,optimize="auto-hq",backend=None,): 
    # Adapts local_expectation from Quimb.tensor.circuit.Circuit
    # Instead of generating rho=ket_1><bra_1 from a given ket_1>, it generates ket_1><bra_2 from two different states.
    # Then it contracts equivalent indices and places the observable G in front to do the final contraction
    # keep indicates on which indices G acts to save computation time

    if isinstance(where, numbers.Integral):
        where = (where,)

    fs_opts = {
        "seq": "ADCRS",
        "atol": 1e-12,
        "equalize_norms": False,
    }

    # rho = ket.get_rdm_lightcone_simplified(where=where, **fs_opts)
    p_bra = bra.copy()
    p_bra.reindex_sites_("b{}", where=where)
    rho = ket.psi & p_bra.H

    k_inds = tuple(ket.ket_site_ind(i) for i in where)
    b_inds = tuple(ket.bra_site_ind(i) for i in where)

    if isinstance(G, (list, tuple)):
        # if we have multiple expectations create an extra indexed stack
        nG = len(G)
        G_data = do("stack", G)
        G_data = autoray.reshape(G_data, (nG,) + (2,) * 2 * len(where))
        output_inds = (qu.tensor.tensor_core.rand_uuid(),)
    else:
        G_data = autoray.reshape(G, (2,) * 2 * len(where))
        output_inds = ()

    TG = qu.tensor.Tensor(data=G_data, inds=output_inds + b_inds + k_inds)

    rhoG = rho | TG

    rhoG.full_simplify_(output_inds=output_inds, **fs_opts)
    # rhoG.astype_("complex128")

    # if rehearse == "tn":
    #     return rhoG

    tree = rhoG.contraction_tree(
        output_inds=output_inds, optimize=optimize
    )

    # if rehearse:
    #     return qu.tensor.circuit.rehearsal_dict(rhoG, tree)

    g_ex = rhoG.contract(
        all,
        output_inds=output_inds,
        optimize=tree,
        backend=backend,
    )

    if isinstance(g_ex, qu.tensor.Tensor):
        g_ex = tuple(g_ex.data)

    return g_ex




def n_half_pis(param) -> int:
    try:
        param = float(param)
        epsilon = (abs(param) + 0.5 * 1e-10) % (np.pi / 2)
        if epsilon > 1e-10:
            raise ValueError(f"{param} is not to a multiple of pi/2")
        multiple = int(np.round(param / (np.pi / 2)))
        return multiple % 4
    except TypeError as err:
        raise ValueError(f"{param} is not bounded") from err
    


def phase_convert(phase):
    if phase==1:
        return '+'
    elif phase==-1:
        return '-'
    elif phase==1j:
        return '+i'
    elif phase==-1j:
        return '-i'



def quimb_inital_state(binary_str):
    quimb_c = qu.tensor.Circuit(len(binary_str))
    for i,ch in enumerate(binary_str):
        if ch=='1':
            quimb_c.apply_gate('X',i)
    return quimb_c



# This condenses a method used below (which does not use this function because it's more complex)
def trigonometrize(vector):
    # For an n-dim vector v, finds an n-dim vector t of angles such that the original vector fulfills:
    # v = ( sin(t1)cos(t2), sin(t1)sin(t2)cos(t3), ... , sin(t1)sin(t2)...sin(tn) )
    # which means that t1 will always be pi/2 and is there only to help with automatization
    vector_trig = []
    for i in range(len(vector)-1):
        coef = vector[i]
        for v in vector_trig:
            coef /= np.sin(v)
        vector_trig.append(np.arccos(coef))

    return vector_trig




def check_complexity(gen_clifford,qubits): 
    # !!!!!!!!!!!!! work in progress !!!!!!!!!!!!!
    if gen_clifford.mode=='dict':
        return 0,gen_clifford
    elif gen_clifford.mode in ['sparse, sparse_comp']:
        return 0,gen_clifford
    elif gen_clifford.mode=='tn':
        old_bond = gen_clifford.xvec.bond_size(qubits[0],qubits[1])
        new_gen_clifford = gen_clifford.copy()
        new_gen_clifford.xvec.gate_(CNOT,(qubits[0],qubits[1]),contract='swap+split')

        new_bond = new_gen_clifford.xvec.bond_size(qubits[0],qubits[1])
        if new_bond > old_bond:
            return 0,gen_clifford
        else:
            return 1,new_gen_clifford
    
        # return gen_clifford.xvec.contraction_width(optimize='random-greedy')
    else: return 0,gen_clifford



        ########### Translation functions ###########

def convert(a,b):
    # Returns the sum of a and b as a binary string. 
    # Inputs are expected as integers or binary strings ('10001011') and converted so that they can be added together.
    try:    
        if isinstance(a, (int, np.integer)):
            a = np.array([int(t) for t in bin(a)[2:]])
        else:
            a = np.array(a)
    except TypeError:
        a = np.array(a)

    try:
        if isinstance(b, (int, np.integer)):
            b = np.array([int(t) for t in bin(b)[2:]])
        else:
            b = np.array(b)
    except TypeError:
        b = np.array(b)

    # make them equal length
    padding = [0,]*np.abs(len(a)-len(b))
    if len(a)<len(b):
        a = np.concatenate((padding,a))
    elif len(a)>len(b):
        b = np.concatenate((padding,b))

    res = []
    for i,j in zip(a,b):
        res.append((i+j)%2)
    res = np.array(res, dtype=str)

    return int(''.join(res),2)




def trans_pauli(observable,qubits=None,total_qubits=None): 
    # Translates an observable in pauli basis from a string of pauli symbols to the boolean clifford basis
    # (in the current implementation the phase of the observable must be handled separately!) 
    if qubits is not None:
        new_obs = ''
        if total_qubits is not None:
            total_qubits = qubits[-1]
        for qb in range(total_qubits):
            if qb in qubits:
                new_obs += observable[qubits.index(qb)]
            else:
                new_obs += 'I'
        observable = new_obs
    else:
        total_qubits = len(observable)

    pauli_array = [0,]*(2*total_qubits)
    trans = {'I': (0,0), 'X': (1,0), 'Y':(1,1), 'Z':(0,1)}
    for i,pauli in enumerate(observable):
        pauli_array[i]=trans[pauli][0]
        pauli_array[i+total_qubits]=trans[pauli][1]

    pauli_array += [0] # add phase

    return pauli_array



def trans_pauli_rev(observable):
    # Translates an observable in pauli basis from the boolean clifford basis to a string of pauli symbols
    # (in the current implementation the phase of the observable must be handled separately!)
    num_qubits = len(observable)//2
    rev_observable = '' # str((-1)**observable[-1]) # first character is the phase of the observable, stored at the end
    trans = {00: 'I', 10: 'X', 11: 'Y', 1: 'Z'} # table of translation with the x,z vectors
    for i in range(num_qubits):
        rev_observable += trans[int(str(int(observable[i])) + str(int(observable[i+num_qubits])))] # p_i is based on x_i, z_i

    return rev_observable


def obs_to_tn(obs,full=False):
    if full: # we can opt to save the whole n-qubit observable but with quimb is not necessary
        expec = qu.pauli(obs[0])
        for i,ch in enumerate(obs[1:]): 
            expec = expec & qu.pauli(ch)
        return expec, []
    else:
        expec = qu.pauli('I') # just in case all obs are "I"
        where = [0,]
        for i,ch in enumerate(obs):
            if ch!='I':
                expec = qu.pauli(ch) # If we find one that isn't we replace it
                where = [i,] # and mark where because TN is then more efficient
                break
        for j,ch in enumerate(obs[i+1:]): # We continue adding the rest if there are more
            if ch!='I':
                expec = expec & qu.pauli(ch)
                where.append(j+1)
        return expec, where


        ########## gate decomposition functions #############


def gate_decomposition(tableau,gate,qubits=None):
    # decomposes a gate in boolean pauli form into the boolean clifford basis
    if type(qubits) is int: qubits = [qubits]
    num_qubits = len(tableau)//2
    destab_v = [0,]*num_qubits
    stab_v = [0,]*num_qubits

    if gate == [0,]*len(gate):
        return 1,destab_v,stab_v

    # We keep track of the operators in the decomposition to find the extra phase needed
    accum = [0,]*len(tableau) + [1,] #the last element is where we will store the phase (like the tableau but complex)
    # checks if it commutes with the destabilizers
    for i in range(len(tableau)//2):
        destab = tableau[i]
        stab = tableau[i+num_qubits]
        stab_v[i],accum = check_comm(gate,destab,stab,accum,qubits) # if it anticommutes, it means stab_v[i] is needed!

    # checks if it commutes with the stabilizer
    for i in range(len(tableau)//2): # we need to do this after doing all stabilizers to get correct phase
        destab = tableau[i]
        stab = tableau[i+num_qubits]
        destab_v[i],accum = check_comm(gate,stab,destab,accum,qubits) # if it anticommutes, it means destab_v[i] is needed!

    phase = accum[-1]

    return phase,destab_v,stab_v




def tgate_decomp(tableau,qubit,dag=False):
    # decomposes the tgate into boolean pauli form 
    gate_list = ([0,0],[0,1])
    gate_coefs = [np.cos(np.pi/8),-1j*np.sin(np.pi/8)]
    if dag: gate_coefs[1]*= -1

    destab_list = []
    stab_list = []
    tot_qubits = len(tableau)//2

    for i,gate in enumerate(gate_list):
        gate_vector = [0,]*(len(tableau))
        gate_vector[qubit] = gate[0]
        gate_vector[qubit+tot_qubits] = gate[1]
        
        phase, destab, stab = gate_decomposition(tableau,gate_vector,qubit)

        gate_coefs[i] *= phase
        destab_list.append(destab)
        stab_list.append(stab)

    return gate_coefs, destab_list, stab_list

def ccz_decomp(tableau,qubits):
    # double check if its x1x2x3z1z2z3 (current) or x1z1x2z2x3z3
    gate_list = [[0,0,0,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,1,1,0],[0,0,0,1,0,1],[0,0,0,0,1,1],[0,0,0,1,1,1]]
    # General coefficients fixing the phase of Id to 1
    gate_coefs = [3/4,1/4,1/4,1/4,-1/4,-1/4,-1/4,1/4]

    final_coefs = []
    destab_list = []
    stab_list = []
    tot_qubits = len(tableau)//2

    for i,gate in enumerate(gate_list):
        gate_vector = [0,]*(len(tableau))
        for i,qubit in enumerate(qubits):
            gate_vector[qubits] = gate[i]
            gate_vector[qubit+tot_qubits] = gate[i+3]

        phase, destab, stab = gate_decomposition(tableau,gate_vector,qubits)

        final_coefs.append(gate_coefs[i]*phase)
        destab_list.append(destab)
        stab_list.append(stab)

    return final_coefs, destab_list, stab_list

def ugate_decomp(tableau,qubit,theta,phi,lambd):
    # decomposes a generic ugate into boolean pauli form 
    gate_list = [[0,0],[1,0],[1,1],[0,1]]
    # General coefficients fixing the phase of Id to 1
    gate_coefs = [np.cos(theta/2)*np.sqrt((1+np.cos(phi+lambd))/2),
                   1j*np.sin(theta/2)*(np.sin(phi)-np.sin(lambd))/np.sqrt(2*(1+np.cos(phi+lambd))),
                   -1j*np.sin(theta/2)*(np.cos(phi)+np.cos(lambd))/np.sqrt(2*(1+np.cos(phi+lambd))),
                   -1j*np.cos(theta/2)*(np.sin(phi+lambd))/np.sqrt(2*(1+np.cos(phi+lambd))),]
    # results from tracing out tr(UX),tr(UY) and tr(UZ)
    # gate_coefs = [np.cos(theta/2)*(1+np.exp(1j*(phi+lambd)))/2,
    #                np.sin(theta/2)*(np.exp(1j*(phi))-np.exp(1j*(lambd)))/2,
    #                -1j*np.sin(theta/2)*(np.exp(1j*(phi))+np.exp(1j*(lambd)))/2,
    #                np.cos(theta/2)*(1-np.exp(1j*(phi+lambd)))/2,]

    final_coefs = []
    destab_list = []
    stab_list = []
    tot_qubits = len(tableau)//2

    for i,gate in enumerate(gate_list):
        if np.abs(gate_coefs[i]) <= 1e-10: 
            continue
        gate_vector = [0,]*(len(tableau))
        gate_vector[qubit] = gate[0]
        gate_vector[qubit+tot_qubits] = gate[1]

        phase, destab, stab = gate_decomposition(tableau,gate_vector,qubit)

        final_coefs.append(gate_coefs[i]*phase)
        destab_list.append(destab)
        stab_list.append(stab)

    return final_coefs, destab_list, stab_list




def cc_gate(qubits,inds,type='x'):
    # decomposes a ccx gate into 1qb and 2qb gates
    temp = QuantumCircuit(qubits)
    if type == 'x':
        temp.h(inds[2])
    elif type == 'y':
        temp.rx(np.pi/2,inds[2])
    elif type != 'z':
        raise CircuitError('cc_gate type not implemented')
    
    temp.cnot(inds[1],inds[2])
    temp.tdg(inds[2])
    temp.cnot(inds[0],inds[2])
    temp.t(inds[2])
    temp.cnot(inds[1],inds[2])
    temp.tdg(inds[2])
    temp.cnot(inds[0],inds[2])
    temp.t([inds[1],inds[2]])
    temp.cnot(inds[0],inds[1])
    temp.t(inds[0])
    temp.tdg(inds[1])
    temp.cnot(inds[0],inds[1])

    if type == 'x':
        temp.h(inds[2])
    elif type == 'y':
        temp.rx(-np.pi/2,inds[2])
    elif type != 'z':
        raise CircuitError('cc_gate type not implemented')

    return temp





################################# Generalized Clifford class ######################################

class gen_clifford(Clifford):
    # To make it easy we only initialize with clifford circuits so we can keep the init

    def __init__(self, data, copy=True, mode='sparse_comp', max_bond=None, cc_direct=False, contract=False, debug=False, *args, **kwargs):
        super(gen_clifford, self).__init__(data, copy=True, *args, **kwargs)

        if isinstance(data, gen_clifford) and copy:
            self._xvec = data.xvec.copy()
            self._mode = data._mode
            self._results = data._results
            self._num_clbits = data.num_qubits
            self._max_bond = data.max_bond
            self._debug = data._debug
            self.cc_direct = data.cc_direct   # Try implementation of cc gate directly
            self._contract = data._contract

            return 

        # initalize bond_matrix if it's not a copy
        if mode=='tn':
            psi0 = qtn.MPS_computational_state('0' * self.num_qubits)
            self._xvec = psi0
        elif mode=='dict':
            self._xvec = {np.array([0])[0]: 1}
        elif mode in ['sparse','sparse_comp']:
            if mode=='sparse_comp':
                xvec = lil_array((1,1))
            else:
                xvec = lil_array((1,2**self.num_qubits),dtype=complex)
            xvec[0,0] = 1
            self._xvec = xvec
        else:
            raise QiskitError('xvec was not initialized')
        
        # store mode for the update method
        self._mode = mode
        self._num_clbits = data.num_qubits
        self._results = {}
        self._max_bond = max_bond # this is useless if mode != 'tn' but it's easier to have the parameter
        self._debug = debug
        self.cc_direct = cc_direct 
        if contract:
            contract = 'swap+split'
        self._contract = contract

    @property
    def xvec(self):
        return self._xvec

    @property
    def mode(self):
        return self._mode

    @property
    def num_clbits(self):
        return self._num_clbits

    @property
    def results(self):
        return self._results
    
    @property
    def max_bond(self):
        return self._max_bond
    
    @property
    def tableau_ordered(self):
        qbs = self.num_qubits
        return [np.concatenate([row[qbs-1::-1],row[2*qbs-1:qbs-1:-1],row[-1:]])
                 for row in self.tableau]
    
    def reduce_bond_dim(self,max_bond=None):
        if max_bond is not None:
            self._max_bond = max_bond
        else:
            max_bond = self.max_bond
        
        if self._mode=='tn':
            if self._contract==False:
                self._xvec.contract(...,max_bond=self.max_bond)
            else:
                self._xvec.compress(max_bond=max_bond)
        else:
            print(f"Mode {self._mode} does not use bond dimension")
            
        return
    

    def to_pure_mps(self):
        # This converts to computational basis (traditional MPS) in a sort of optimal way.
        # to_quimb_circuit uses qiskit's to_circuit to extract a Clifford circuit from the current tableau (optimal in depth)
        # Then it can apply this circuit on to the MPS in tensor network form by choosing on_mps=True
        return self.to_quimb_circuit(on_mps=True).contract(...,max_bond=self.max_bond)


    def computational_basis(self,tol=1e-10j):
        # this is brute force, there might be a better way to do it!
        qubits = self.num_qubits
        comp_vec = np.zeros(2**qubits,dtype=complex)
        format_s = '{'+f":0>{qubits}b"+'}'
        stab_ket = self.to_quimb_circuit()

        for i in range(2**qubits):
            # bra_qc = quimb_inital_state(format_s.format(i))
            bra = qu.tensor.tensor_builder.MPS_computational_state(format_s.format(i))
            # print(bra.H @ stab_ket.psi)
            res = 0j
            print(f"checking state {format_s.format(i)}")
            for j in range(2**qubits):
                coef = self.xvec.contract().data[*[int(ch) for ch in format_s.format(j)]] 
                if np.abs(coef)<tol: 
                    continue
                op = [0,]*(2*qubits)+[1]
                # print(j)
                print(format_s.format(j))
                for k in range(qubits):
                    if format_s.format(j)[k] == '1':
                        op = multiply_bool_pauli(op,self.tableau[k])
                phase = op[-1]
                trans_op = trans_pauli_rev(op)
                print(phase_convert(phase)+' '+trans_op)
                expec, where = obs_to_tn(trans_op)
                val = phase * expect_tn(bra,expec,stab_ket,where)
                res += coef * val if np.abs(coef*val) > tol else 0
                print(f"coefficient:{coef}")
                print(f"expected value: {val}")
                print(f"added value: {coef * val}")
                print(f"current res: {res}")
            if i==0:
                glob_phase = np.conj(res)/np.sqrt(res*np.conj(res))

            comp_vec[i] = glob_phase*res

        return comp_vec.reshape([2,]*qubits)
    

    def to_quimb_circuit(self,on_mps=False):
        if on_mps:
            quimb_c = qu.tensor.Circuit(self.num_qubits,self.xvec)
        else:
            quimb_c = qu.tensor.Circuit(self.num_qubits)
        qiskit_c = self.to_circuit()
        for gt in qiskit_c:
            quimb_c.apply_gate(gt.operation.name, *[qiskit_c.find_bit(qb).index for qb in gt.qubits])

        return quimb_c
    


    # we need to change the compose method to work with non-cliffords
    def compose(self,
        other: QuantumCircuit or Instruction,
        qargs: list or None = None,
        front: bool = False,
    ) -> Clifford:
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        # If other is a QuantumCircuit we can more efficiently compose
        # using the _append_circuit method to update each gate recursively
        # to the current Clifford, rather than converting to a Clifford first
        # and then doing the composition of tables.
        if not front:
            if isinstance(other, QuantumCircuit):
                self._append_gen_circuit(other, qargs=qargs)
            if isinstance(other, Instruction):
                self._append_gen_operation(other, qargs=qargs)

        return self

    def measure_obs(self, observable, qubits=None):
        if type(observable) is str:
            observable_v = trans_pauli(observable)
        else:
            observable_v = observable
            observable = trans_pauli_rev(observable)

        self.measure(observable_v,observable,qubits)

        return self._results
    
    def project_obs(self, observable, qubits=None):
        if type(observable) is str:
            observable_v = trans_pauli(observable)
        else:
            observable_v = observable
            observable = trans_pauli_rev(observable)

        self.measure(observable_v,observable,qubits,project=True)

        return self._results

    def meas_tableau(self, observable, destab, stab, sign):
        # modifies the tableau once we have measured a specific observable
        tableau = self.tableau.copy()
        
        k = destab.index(1)
        qubits = len(tableau)//2
        stab_k = tableau[k+qubits]

        for i,b in enumerate(destab):
            if b:
                tableau[i+qubits] = [(tableau[i+qubits][j] + stab_k[j])%2 for j in range(len(stab_k))]
        for i,c in enumerate(stab):
            if i==k:
                tableau[i] = stab_k
            if c:
                tableau[i] = [(tableau[i][j] + stab_k[j])%2 for j in range(len(stab_k))]

        tableau[k+qubits] = observable

        if sign<0:
            tableau[k+qubits][-1] = 1

        self.tableau = tableau
        return tableau

    def normalize(self, insert=-1):
        # Normalizes a tensor network
        if self.mode != 'tn':
            print("Normalize method was called for non-tn mode")
            return 
        
        tn = self.xvec
        norm = tn.norm()
        tn.tensors[insert].modify(data=tn.tensors[insert].data / norm)

        return tn

    def read_tableau_obs(self,destab,stab):
        # Returns the Pauli operator corresponding to an observable (obs) given 
        # in tableau form, using the current destabilizer basis.
        qubits = self.num_qubits
        pauli_form = [0,]*(2*qubits) + [1,]
        for i,check in enumerate(destab+stab):
            if check: 
                pauli_form = multiply_bool_pauli(pauli_form,self.tableau[i])
        phase = pauli_form[-1]

        return phase, trans_pauli_rev(pauli_form)

    # "read_tableau_obs" can be used like this
    # print(f"coefficients from ugate decomp:")
    # for coef,destab,stab in zip(gate_coefs,destab_list,stab_list):
    #     print(f"coeff: {coef}, operator: {self.read_tableau_obs(destab,stab)}")

    def apply_xvec_rot(self,angle,ind_dict,contract=False):
        # For a given d we need to implement a R[(X/Y/Z)_i] for all qubits i involved in d and s
        # This is done with a cascade of CNOTS, an RX and extra 1qb transf [arxiv/2305.04807]
        if contract: contract='swap+split'
        diff_inds = [ind for ind in ind_dict]

        rot_ind = int(len(diff_inds)/2)
        rot_qubit = diff_inds[rot_ind]

        for j in ind_dict:
            if ind_dict[j]=='Y':
                self._xvec.gate_(S,j, contract=True)
            elif ind_dict[j]=='Z':
                self._xvec.gate_(H,j, contract=True)

        prev_ind = diff_inds[0]
        for j in diff_inds[1:rot_ind+1]:
            self._xvec.gate_(CNOT, (j, prev_ind), contract=contract)
            prev_ind = j
        prev_ind = diff_inds[-1]
        for j in diff_inds[-2:rot_ind-1:-1]:
            self._xvec.gate_(CNOT, (j, prev_ind), contract=contract)
            prev_ind = j

        self._xvec.gate_(RX(2*angle), (rot_qubit), contract=True)

        prev_ind = rot_qubit
        for j in diff_inds[rot_ind-1::-1]:
            if rot_ind == 0:
                continue
            self._xvec.gate_(CNOT, (prev_ind,j), contract=contract)
            prev_ind = j
        prev_ind = rot_qubit
        for j in diff_inds[rot_ind+1:]:
            self._xvec.gate_(CNOT, (prev_ind,j), contract=contract)
            prev_ind = j

        for j in ind_dict:
            if ind_dict[j]=='Y':
                self._xvec.gate_(Sdg,j, contract=True)
            elif ind_dict[j]=='Z':
                self._xvec.gate_(H,j, contract=True)


    def update_xvec(self,coefs,destab_list,stab_list,tolerance=1e-10):
        # Main method to update xvec. Different applications depending on the format of the vector
        mode = self.mode
        contract = self._contract

        if mode == 'tn':
            params_sort = sorted(zip(coefs,destab_list,stab_list), key=lambda ins: sum(ins[1])) # [ordered array of (coef,destab,stab)]
            destab_ref = params_sort[0][1]
            for i,entry in enumerate(destab_ref):
                if entry: self._xvec.gate_(X,i,contract=True) # ,contract='swap+split' # apply gates to qubits where the first destabilizer is not 0s
            stab_ref = params_sort[0][2]
            for i,entry in enumerate(stab_ref):
                if entry: self._xvec.gate_(Z,i,contract=True) # ,contract='swap+split' # apply gates to qubits where the first stabilizer is not 0s    
            total_coef = params_sort[0][0]
            angles = []
            ind_dicts = []
            if len(params_sort) > 2:
                print(f"applying decomposition with {len(params_sort)} terms")
            for (co,d,s) in params_sort[1:]:
                # Prepare the angles and the axes for the proper rotations based on the decomposition into stabs/destabs
                d_differential = [(destab_ref[i]+dest)%2 for i,dest in enumerate(d)] 
                s_differential = [(stab_ref[i]+st)%2 for i,st in enumerate(s)] 
                extra_sign = (-1)**(sum(np.array(stab_ref)*np.array(d)))

                ind_dict = {}
                Ys = 0        
                for j in range(len(d_differential)):
                    if d_differential[j]:
                        if s_differential[j]:
                            ind_dict[j] = 'Y'
                            Ys += 1
                        else:
                            ind_dict[j] = 'X'
                    elif s_differential[j]:
                        ind_dict[j] = 'Z'

                ind_dicts.append(ind_dict)

                total_coef = np.sqrt(total_coef**2 + np.abs(co)**2)
                co *= np.conj((-1j)**Ys) * 1j # extract -1js from Ys in exponential and -1j from rotation
                phase = co/np.abs(co) # this should have the correct sign after extracting 1j components

                # sanity checks ########### ALL THESE SHOULD BE CONVERTED TO RAISED ERRORS ###########
                # 1 : Coefficients bigger than 1
                if (np.abs(co)-1)>1e-8: 
                    print('Found coefficient bigger than 1 by more than 1e-8. Unlikely to be numerical: recheck calculation.')
                elif (np.abs(co)-1)>0:
                    co = np.sign(co)*1
                # 2 : after extracting all the 1j factors phase can only be 1 or -1
                if np.imag(phase)!=0 : print(f"Something went wrong with the angles! Phase is {phase}")        

                angles.append(extra_sign*np.arcsin(co/total_coef))

            # Apply rotations following [arxiv/1907.09040] for the implementation of a unitary decomposed into several Paulis
            for i in range(len(params_sort)-2):
                self.apply_xvec_rot(angles[i]/2,ind_dicts[i],contract=contract) 
            self.apply_xvec_rot(angles[-1],ind_dicts[-1],contract=contract)
            for i in range(len(params_sort)-2)[::-1]:
                self.apply_xvec_rot(angles[i]/2,ind_dicts[i],contract=contract) 

            if contract:
                self._xvec.compress(max_bond=self.max_bond)

            if self._debug:
                print('xvec updated')
                print(self._xvec)
                    
        elif mode in ['sparse','sparse_comp']:

            _, cols = self._xvec.nonzero()

            if mode == 'sparse':
                new_xvec = lil_array(self._xvec.shape,dtype=complex)
            elif mode == 'sparse_comp':
                # try to make it as big as possible. This will usually be enough without an exhaustive search
                new_xvec = lil_array((self._xvec.shape[0],max([self._xvec.shape[1],]+[convert(cols[-1],d)+1 for d in destab_list])),dtype=complex)
            
            for co,d,s in zip(coefs,destab_list,stab_list):
                if np.abs(co)<tolerance:
                    continue
                for c in cols:
                    c_bin = np.array([t for t in format(c, '0' + str(len(s)) + 'b')],dtype=int)
                    if convert(c,d)>=new_xvec.shape[1]:
                        # if the method above comes short, this will fix it
                        expanded_xvec = lil_array((1,convert(c,d)+1),dtype=complex)
                        _,cols_bis = new_xvec.nonzero()
                        for cbis in cols_bis:
                            expanded_xvec[(0,cbis)] = new_xvec[(0,cbis)]
                        new_xvec = expanded_xvec
                    res = new_xvec[0,convert(c,d)] + co*(-1)**(sum(np.array(s)*np.array(c_bin))) * self._xvec[0,c]
                    if np.abs(res)>tolerance:
                        new_xvec[0,convert(c,d)] = res
                    else:
                        new_xvec[0,convert(c,d)] = 0

            self._xvec = new_xvec
        
        elif mode=='dict':

            new_xvec = {}
            cols = [key for key in self._xvec]
            for co,d,s in zip(coefs,destab_list,stab_list):
                if np.abs(co)<tolerance:
                    continue
                for c in cols:
                    c_bin = np.array([t for t in format(c, '0' + str(len(s)) + 'b')],dtype=int)
                    target_ind = convert(c,d)
                    if target_ind not in new_xvec:
                        new_xvec[target_ind] = 0
                    res = new_xvec[convert(c,d)] + co*(-1)**(sum(np.array(s)*np.array(c_bin))) * self._xvec[c]
                    if np.abs(res)>tolerance:
                        new_xvec[convert(c,d)] = res
                    else:
                        new_xvec[convert(c,d)] = 0

            self._xvec = new_xvec
                            
        return self._xvec

    def measure(self,observable,tag,qubits=None,project=False):
        # tag is how we will identify the stored result. 
        # If coming from qiskit, one can just use the clbit that was assigned to that measurement
        tableau = self.tableau
        xvec = self._xvec
        contract = self._contract

        num_qubits = len(tableau)//2
        if type(observable) is str:
            observable_v = trans_pauli(observable)
        elif len(observable)==len(tableau[0]):
            observable_v = observable
        else:
            print("Did not recognize format of observable to measure")
            return {}
        
        if qubits is not None:
            try: len(qubits)>1
            except: qubits = [qubits]

        phase, destab, stab = gate_decomposition(self.tableau,observable_v,qubits=qubits)

        if self.mode == 'tn':
            ev = phase
            new_xvec = xvec.copy()
            ref_xvec = xvec.conj()

            for qb,val in enumerate(stab):
                if val:
                    xvec.gate_(Z,qb)
            for qb,val in enumerate(destab):
                if val:
                    xvec.gate_(X,qb)

            ev *= ref_xvec @ xvec
            ev = np.round(ev,10)

            out0 = (1+ev)/2
            out1 = (1-ev)/2
            outcome = random()>out0

            if project:
            # Projection
                print('Projecting state onto measured observable')
                phase *= (-1)**outcome # takes into account if the result was 0 or 1
                angle = np.pi/4

                ind_dict = {}
                Ys = 0    
                for i,(d,s) in enumerate(zip(destab,stab)):
                    if d:
                        if s:
                            ind_dict[i] = 'Y'
                            Ys += 1
                        else:
                            ind_dict[i] = 'X'
                    elif s:
                        ind_dict[i] = 'Z'

                diff_inds = [ind for ind in ind_dict]
                rot_ind = int(len(diff_inds)/2)
                rot_qubit = diff_inds[rot_ind]

                # basis change
                for i in ind_dict:
                    if ind_dict[i]=='Y':
                        new_xvec.gate_(S,i, contract=True)
                    elif ind_dict[i]=='Z':
                        new_xvec.gate_(H,i, contract=True)

                # CNOTS input
                prev_ind = diff_inds[0]
                for i in diff_inds[1:rot_ind+1]:
                    new_xvec.gate_(CNOT, (i,prev_ind), contract=contract)
                    prev_ind = i
                prev_ind = diff_inds[-1]
                for i in diff_inds[-2:rot_ind-1:-1]:
                    new_xvec.gate_(CNOT, (i, prev_ind), contract=contract)
                    prev_ind = i

                # Core rotations
                renorm = 1/np.sqrt(1+np.abs(ev)) 
                rot_matrix = quimbify([[np.cos(angle)*renorm, phase * (-1)**Ys * np.sin(angle)*renorm],
                            [phase * (-1)**Ys * np.sin(angle)*renorm, np.cos(angle)*renorm]]) # this is non-unitary!!
                new_xvec.gate_(rot_matrix, (rot_qubit), contract=True)

                # CNOTS output
                prev_ind = rot_qubit
                for i in diff_inds[rot_ind-1::-1]:
                    if rot_ind == 0:
                        continue
                    new_xvec.gate_(CNOT, (prev_ind,i), contract=contract)
                    prev_ind = i
                prev_ind = rot_qubit
                for i in diff_inds[rot_ind+1:]:
                    new_xvec.gate_(CNOT, (prev_ind,i), contract=contract)
                    prev_ind = i

                # basis unchange
                for i in ind_dict:
                    if ind_dict[i]=='Y':
                        new_xvec.gate_(Sdg,i, contract=True)
                    elif ind_dict[i]=='Z':
                        new_xvec.gate_(H,i, contract=True)

                # remove the entries for which i·k = 1
                if 1 in destab:
                    k = destab.index(1)
                    new_xvec.gate_(quimbify([[np.sqrt(2),0],[0,0]]), k, contract=True) # this is non-unitary! # it also can be done with a |0> contraction
                    self.tableau = self.meas_tableau(observable_v,destab,stab,(-1)**outcome)

                # despite the non-unitary matrices, renormalization (which is taken into account) should restore phyisicality, 
                # which is ensured here:
                if self._contract==False:
                    new_xvec.contract(...,max_bond=self.max_bond)
                else:
                    new_xvec.compress(max_bond=self.max_bond)
                new_xvec = self.normalize()
                self._xvec = new_xvec

            results = {'reg': int(outcome), 'stats':(out0, out1), 'ev':ev} 
            self._results[tag] = results

            return results
        
        elif self.mode in ['sparse','sparse_comp']:
            _, cols = xvec.nonzero()
            inds = lambda x : (0,x)

            if self.mode == 'sparse':
                new_xvec_0 = lil_array((1,2**num_qubits),dtype=complex)
                new_xvec_1 = lil_array((1,2**num_qubits),dtype=complex)
            elif self.mode == 'sparse_comp':
                shape = (xvec.shape[0],max([xvec.shape[1],]+[convert(c,destab) for c in cols]))
                new_xvec_0 = lil_array(shape,dtype=complex)
                new_xvec_1 = lil_array(shape,dtype=complex)
        elif self.mode=='dict':
            cols = [key for key in xvec]
            inds = lambda x : x

            new_xvec_0 = {}
            new_xvec_1 = {}

        if destab == [0,]*num_qubits:
            out0 = 0
            out1 = 0
            for c in cols:
                c_bin = np.array([t for t in format(c, '0' + str(len(stab)) + 'b')],dtype=int)
                val = xvec[inds(c)]
                if phase*(-1)**(sum(np.array(stab)*np.array(c_bin)))>0:
                    new_xvec_0[inds(c)] = val
                    new_xvec_1[inds(c)] = 0
                    out0 += val * np.conjugate(val)
                elif (-1)*phase*(-1)**(sum(np.array(stab)*np.array(c_bin)))>0:
                    new_xvec_1[inds(c)] = val
                    new_xvec_0[inds(c)] = 0
                    out1 += val * np.conjugate(val)
                else:
                    print('a value was not counted')
            
            tot = out0 + out1
            # safety check
            if np.abs(1-tot)>1e-6:
                print('Measurement outcomes do not sum 1')
                for c in cols:
                    new_xvec_0[inds(c)] /= tot
                    new_xvec_1[inds(c)] /= tot
                out0 /= tot
                out1 /= tot
            outcome = random()>out0
            if outcome:
                new_xvec = new_xvec_1
            else:
                new_xvec = new_xvec_0
            ev = out0-out1
        else:
            k = [0,]*num_qubits
            k[destab.index(1)] = 1
            ev = 0
            renorm_0 = 0
            renorm_1 = 0

            for c in cols:
                coef = 1/np.sqrt(2)
                c_bin = np.array([t for t in format(c, '0' + str(len(stab)) + 'b')],dtype=int)

                if sum(np.array(k)*np.array(c_bin))%2:
                    coef_0 *= phase * (-1)**(sum(np.array(stab)*np.array(c_bin)))
                    coef_1 = -coef_0
                    target_ind = inds(convert(c,destab))
                else:
                    coef_0 = coef
                    coef_1 = coef
                    target_ind = inds(c)

                if self.mode=='dict' and target_ind not in new_xvec_0:
                    new_xvec_0[target_ind] = 0
                    new_xvec_1[target_ind] = 0
                if new_xvec_0[target_ind] != 0:
                    renorm_0 -= new_xvec_0[target_ind] * np.conjugate(new_xvec_0[target_ind])
                    renorm_1 -= new_xvec_1[target_ind] * np.conjugate(new_xvec_1[target_ind])
                new_xvec_0[target_ind] += coef_0*xvec[inds(c)]
                new_xvec_1[target_ind] += coef_1*xvec[inds(c)]
                renorm_0 += new_xvec_0[target_ind] * np.conjugate(new_xvec_0[target_ind])
                renorm_1 += new_xvec_0[target_ind] * np.conjugate(new_xvec_0[target_ind])

                if self.mode=='dict' and inds(convert(c,destab)) not in xvec:
                    conj = 0
                elif self.mode=='sparse_comp' and convert(c,destab)>xvec.shape[1]:
                    conj = 0
                else:
                    conj = np.conjugate(xvec[inds(convert(c,destab))])

                ev += phase * xvec[inds(c)] * conj * (-1)**(sum(np.array(stab)*np.array(c_bin)))

                # sanity check for realness
                if np.abs(np.imag(ev))>1e-12: print('We got a complex expected value!')

            out0 = (1+ev)/2
            out1 = (1-ev)/2
            outcome = random()>out0

            if outcome:
                new_xvec = new_xvec_1
                renorm = renorm_1
            else:
                new_xvec = new_xvec_0
                renorm = renorm_0

            if self.mode=='dict':
                cols = [key for key in new_xvec]
            elif self.mode in ['sparse','sparse_comp']:
                _, cols = new_xvec.nonzero()
            for c in cols:
                new_xvec[inds(c)] *= 1/np.sqrt(renorm)

            self.tableau = self.meas_tableau(observable_v,destab,stab,(-1)**outcome)

        results = {'reg': int(outcome), 'stats':(out0, out1), 'ev':ev} 
        self._results[tag] = results
        self._xvec = new_xvec

        return results


    ####### Modifying this in the original qiskit should help us #######
    # Right now it's better to initialize with an empty circuit and then 
    # use our "compose" method for all gates (including clifford)
    def _append_gen_circuit(self, circuit, qargs=None, cargs=None):
        # Copy of _append_circuit with the _apply_gen_operation below instead (see Qiskit documentation)
        if qargs is None:
            qargs = list(range(self.num_qubits))
        if cargs is None:
            cargs = list(range(self.num_clbits))
        for instruction in circuit:
            # start = time()
            if instruction.clbits and instruction.operation.name!='measure':
                raise QiskitError(
                    f"Cannot apply Instruction with classical bits: {instruction.operation.name}"
                )
            elif instruction.operation.name == 'measure':
                cbit = cargs[circuit.find_bit(instruction.clbits[0]).index] 
                qbit = qargs[circuit.find_bit(instruction.qubits[0]).index]
                observable = [0,]*(2*self.num_qubits+1)
                observable[qbit + self.num_qubits] = 1
                self.measure(observable,f"cbit_{cbit}",qbit)
                continue
            # Get the integer position of the flat register
            new_qubits = [qargs[circuit.find_bit(bit).index] for bit in instruction.qubits]
            self._append_gen_operation(instruction.operation, new_qubits)

        # Sanity check for compression (slows down computation a)
        # if self.mode == 'tn':
        #     self.reduce_bond_dim()
        return

    def _append_gen_operation(self, operation, qargs=None):
        # Modified _append_operation (see Qiskit documentation) to work with general non-clifford gates

        # Basis Clifford Gates
        basis_1q = {"i": self._append_i, "id": self._append_i,"iden": self._append_i,
        "x": self._append_x,"y": self._append_y,"z": self._append_z,"h": self._append_h,
        "s": self._append_s,"sdg": self._append_sdg,"sinv": self._append_sdg,
        "v": self._append_v,"w": self._append_w,}
        basis_2q = {"cx": self._append_cx, "cz": self._append_cz, "swap": self._append_swap}

        # Non-clifford gates
        non_clifford = ["t", "tdg", "ccx", "ccz"]

        if isinstance(operation, (Barrier, Delay)):
            return 

        if qargs is None:
            print('Found gate with undetermined application qubit. Applying to first available qubits.')
            qargs = list(range(self.num_qubits))
            
        gate = operation
        
        if isinstance(gate, str):
            name = gate
        else:
            # assert isinstance(gate, Instruction)
            name = gate.name
            if getattr(gate, "condition", None) is not None:
                raise QiskitError("Conditional gate is not a valid Clifford operation.")

        # Apply gate if it is a Clifford basis gate
        if name in non_clifford:
            if name=='t':   
                gate_coefs, destab_list, stab_list = tgate_decomp(self.tableau,qargs[0])
                self.update_xvec(gate_coefs, destab_list, stab_list)
            if name=='tdg':
                gate_coefs, destab_list, stab_list = tgate_decomp(self.tableau,qargs[0],dag=True)
                self.update_xvec(gate_coefs, destab_list, stab_list)
            if name=='ccx':
                if self.cc_direct:
                    self._append_h(qargs[2])
                    gate_coefs, destab_list, stab_list = ccz_decomp(self.tableau,qargs[0:2])
                    self.update_xvec(gate_coefs, destab_list, stab_list)
                    self._append_h(qargs[2])
                else:
                    temp = cc_gate(self.num_qubits,qargs[:3],type='x')
                    self._append_gen_circuit(temp)
            if name=='ccz':
                if self.cc_direct:
                    gate_coefs, destab_list, stab_list = ccz_decomp(self.tableau,qargs[0:2])
                    self.update_xvec(gate_coefs, destab_list, stab_list)
                else:
                    temp = cc_gate(self.num_qubits,qargs[:3],type='z')
                    self._append_gen_circuit(temp)
            return
        
        if name in basis_1q:
            if len(qargs) != 1:
                raise QiskitError("Invalid qubits for 1-qubit gate.")
            basis_1q[name](qargs[0])
            return 
        if name in basis_2q:
            if len(qargs) != 2:
                raise QiskitError("Invalid qubits for 2-qubit gate.")
            # if name=='cx': ########################################################## work in progress
            #     cheaper, new_gen_clifford = check_complexity(gen_clifford,qargs)
            #     if cheaper:
            #         return new_gen_clifford
            basis_2q[name](qargs[0], qargs[1])
            return

        # If u gate, check if it is a Clifford, and if so, apply it
        if isinstance(gate, Gate) and name == "u" and len(qargs) == 1:
            try:
                theta, phi, lambd = tuple(n_half_pis(par) for par in gate.params)
            except ValueError as err:
                theta, phi, lambd = tuple(par for par in gate.params)
                gate_coefs, destab_list, stab_list = ugate_decomp(self.tableau,qargs[0],theta,phi,lambd)
                self.update_xvec(gate_coefs, destab_list, stab_list)
            if theta == 0:
                self._append_rz(qargs[0], lambd + phi)
            elif theta == 1:
                self._append_rz(qargs[0], lambd - 2)
                self._append_h(qargs[0])
                self._append_rz(qargs[0], phi)
            elif theta == 2:
                self._append_rz(qargs[0], lambd - 1)
                self._append_x(qargs[0])
                self._append_rz(qargs[0], phi + 1)
            elif theta == 3:
                self._append_rz(qargs[0], lambd)
                self._append_h(qargs[0])
                self._append_rz(qargs[0], phi + 2)
            return 

        # If gate is a Clifford, we can either unroll the gate using the "to_circuit"
        # method, or we can compose the Cliffords directly. Experimentally, for large
        # cliffords the second method is considerably faster.

        if isinstance(gate,Gate) and name in ['rx','ry','rz'] and len(qargs) == 1: 
            try:
                theta = n_half_pis(gate.params[0])
                if name=='rz':
                    self._append_rz(qargs[0], theta)
                elif name=='rx':
                    self._append_h(qargs[0])
                    self._append_rz(qargs[0], theta)
                    self._append_h(qargs[0])
                elif name=='ry':
                    self._append_sdg(qargs[0])
                    self._append_h(qargs[0])
                    self._append_rz(qargs[0], theta)
                    self._append_h(qargs[0])
                    self._append_s(qargs[0])
                
                return gen_clifford
            except ValueError as err:
                theta = gate.params[0]
                if name=='rz':
                    theta, phi, lambd = 0, 0, theta
                elif name=='rx':
                    theta, phi, lambd = theta, -np.pi/2, np.pi/2
                elif name=='ry':
                    theta, phi, lambd = theta, 0, 0
                gate_coefs, destab_list, stab_list = ugate_decomp(self.tableau,qargs[0],theta,phi,lambd)
                # This correction to match usual RZ definition is unnecessary computationally
                # if name=='rz':
                #     gate_coefs = [t*np.exp(-1j*theta/2) for t in gate_coefs]
                self.update_xvec(gate_coefs, destab_list, stab_list)
                return 

        if isinstance(gate, Clifford):
            composed_clifford = self.compose(gate, qargs=qargs, front=False)
            self.tableau = composed_clifford.tableau
            return 
            

        # If the gate is not directly appendable, we try to unroll the gate with its definition.
        # This succeeds only if the gate has all-Clifford definition (decomposition).
        # If fails, we need to restore the clifford that was before attempting to unroll and append.
        if gate.definition is not None:
            try:
                self._append_gen_circuit(gate.definition, qargs) 
                return
            except QiskitError:
                pass

        # As a final attempt, if the gate is up to 3 qubits,
        # we try to construct a Clifford to be appended from its matrix representation.
        if isinstance(gate, Gate) and len(qargs) <= 3:
            try:
                matrix = gate.to_matrix()
                gate_cliff = Clifford.from_matrix(matrix)
                self._append_gen_operation(gate_cliff, qargs=qargs)
                return
            except TypeError as err:
                raise QiskitError(f"Cannot apply {gate.name} gate with unbounded parameters") from err
            except CircuitError as err:
                raise QiskitError(f"Cannot apply {gate.name} gate without to_matrix defined") from err
            except QiskitError as err:
                raise QiskitError(f"Cannot apply non-Clifford gate: {gate.name}") from err

        raise QiskitError(f"Cannot apply {gate}")

    ######## For all the following _append_gate's we have: ########
    """Apply *arbitrary gate* to a Clifford.

        Args:
            clifford (Clifford): a Clifford.
            qubit (int): gate qubit index.

        Returns:
            Clifford: the updated Clifford.
    """
    def _append_i(self, qubit):
        # Apply an I gate to a Clifford.
        return

    def _append_x(self, qubit):
        # Apply an X gate to a Clifford.
        self.phase ^= self.z[:, qubit]
        return 

    def _append_y(self, qubit):
        # Apply a Y gate to a Clifford.
        x = self.x[:, qubit]
        z = self.z[:, qubit]
        self.phase ^= x ^ z
        return

    def _append_z(self, qubit):
        # Apply an Z gate to a Clifford.
        self.phase ^= self.x[:, qubit]
        return 

    def _append_rz(self, qubit, multiple):
        # Apply an Rz gate to a Clifford.
        if multiple % 4 == 1:
            self._append_s(qubit)
        if multiple % 4 == 2:
            self._append_z(qubit)
        if multiple % 4 == 3:
            self._append_sdg(qubit)
        return

    def _append_h(self, qubit):
        # Apply a H gate to a Clifford.
        x = self.x[:, qubit]
        z = self.z[:, qubit]
        self.phase ^= x & z
        tmp = x.copy()
        x[:] = z
        z[:] = tmp
        return 

    def _append_s(self, qubit):
        # Apply an S gate to a Clifford.
        x = self.x[:, qubit]
        z = self.z[:, qubit]
        self.phase ^= x & z
        z ^= x
        return

    def _append_sdg(self, qubit):
        # Apply an Sdg gate to a Clifford.
        x = self.x[:, qubit]
        z = self.z[:, qubit]
        self.phase ^= x & ~z
        z ^= x
        return 

    def _append_v(self, qubit):
        # Apply a V gate to a Clifford.
        x = self.x[:, qubit]
        z = self.z[:, qubit]
        tmp = x.copy()
        x ^= z
        z[:] = tmp
        return

    def _append_w(self, qubit):
        # Apply a W gate to a Clifford.
        x = self.x[:, qubit]
        z = self.z[:, qubit]
        tmp = z.copy()
        z ^= x
        x[:] = tmp
        return 

    def _append_cx(self, control, target):
        # Apply a CX gate to a Clifford.
        x0 = self.x[:, control]
        z0 = self.z[:, control]
        x1 = self.x[:, target]
        z1 = self.z[:, target]
        self.phase ^= (x1 ^ z0 ^ True) & z1 & x0
        x1 ^= x0
        z0 ^= z1
        return 

    def _append_cz(self, control, target):
        # Apply a CZ gate to a Clifford.
        x0 = self.x[:, control]
        z0 = self.z[:, control]
        x1 = self.x[:, target]
        z1 = self.z[:, target]
        self.phase ^= x0 & x1 & (z0 ^ z1)
        z1 ^= x0
        z0 ^= x1
        return

    def _append_swap(self, qubit0, qubit1):
        # Apply a Swap gate to a Clifford.
        self.x[:, [qubit0, qubit1]] = self.x[:, [qubit1, qubit0]]
        self.z[:, [qubit0, qubit1]] = self.z[:, [qubit1, qubit0]]
        return 
        

    ######### Work in progress ##########
    # def reduce_distance(self,override=False): # this method is only a performance boost for TN mode
        
    #     # Checks current tableau to see if there is a simplification that makes pauli matrices X,Y,Z 
    #     # more local in terms of index i weight (how many entries are different)

    #     self.tableau
    #     # should be ran after every update for general simplifications (i.e. guaranteed to improve) and also
    #     # after knowing which qubit to apply a specific transformation (i.e. better for some but worse for others)

    #     return self.tableau