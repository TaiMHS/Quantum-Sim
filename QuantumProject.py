import qiskit, os
import numpy as np
import copy
import time
from qiskit import QuantumCircuit
from qiskit.circuit.library import *

# This Python File is created by Tai Nguyen
# Latest edit date: 4/21/2025
# Intended for FRI Spring 2025

# Globals (potentially used for searching specific states)
max_shots = 32768 # Maximum shots for the simulator
quantum_circuit: QuantumCircuit = None
statevector = None # Statevector of the circuit
deci = None # Decimal places for rounding


# Used for measurement operations
measure_op_found: bool = False # Used to check if measure operation is found in the file
r_measure = [] # Measured states for randomization, serves as all visualization seeds
blank = []

qc_tuplelist = [] # list of tuples of measured qubits / bits
# (qubit, classical bit, measurement set)

random_states = 100
# This is the number of random states to be generated for the measurement operation
# Basically visualization seeds

for i in range(random_states): # Creates blank states to store measurement operations
    r_measure.append(blank)

altered_vector_state = [] # Used to store the altered statevector for measurement operations
measure_sets = 0 # Counts grouped measurements before other gates are applied afterwards
measure_last_op = False # Checks last operation if its a measurement operation

def reset_globals():
    global random_states
    global blank

    # Ones to reset
    global statevector
    global deci
    global measure_last_op
    global measure_op_found
    global r_measure
    global qc_tuplelist
    global altered_vector_state
    global measure_sets

    statevector = None
    deci = None
    measure_last_op = False
    measure_op_found = False
    r_measure = []
    qc_tuplelist = []
    altered_vector_state = []
    measure_sets = 0

    for i in range(random_states):
        r_measure.append(blank)




# To-do
# Make a method that performs the right gates to the right qubit (done, used kronecker products)
# Make a method that computes the state before measurement (done, statevector made)
# make a counts input and use it to create shots
# Perform proper state vector calcs

# Other stuff
# Make a text display of results |More to do|
# Implement measure operation:


# For single qubit gates
def grab_gate(ins_name: str, params):
    if ins_name in "id iden": # Identity gate
        return np.array([[1+0j, 0+0j],
                         [0+0j, 1+0j]], dtype=complex)
    elif ins_name == "x": # Pauli x gate
        return np.array([[0+0j, 1+0j],
                         [1+0j, 0+0j]], dtype=complex)
    elif ins_name == "y": # Pauli y gate
        return np.array([[0+0j, 0-1j],
                         [0+1j, 0+0j]], dtype=complex)
    elif ins_name == "z": # Pauli z gate
        return np.array([[1+0j, 0+0j],
                         [0+0j, -1+0j]], dtype=complex)
    elif ins_name == "h": # Hadamard gate
        return (1/np.sqrt(2)) * np.array([[1+0j, 1+0j],
                                          [1+0j, -1+0j]], dtype=complex)
    elif ins_name == "s": # S gate
        return np.array([[1+0j, 0+0j],
                         [0+0j, 0+1j]], dtype=complex)
    elif ins_name == "sdg": # S dagger gate
        return np.array([[1+0j, 0+0j],
                         [0+0j, 0-1j]], dtype=complex)
    elif ins_name == "s": # S gate
        return np.array([[1+0j, 0+0j],
                         [0+0j, 0+1j]], dtype=complex)
    elif ins_name == "t": # T gate
        return np.array([[1 + 0j, 0 + 0j],
                         [0 + 0j, np.exp(1j * np.pi / 4)]], dtype=complex)
    elif ins_name == "tdg": # T dagger gate
        return np.array([[1 + 0j, 0 + 0j],
                         [0 + 0j, np.exp(-1j * np.pi / 4)]], dtype=complex)
    elif ins_name in  "u u3": # General Unitary gate
        return np.array([[np.cos(params[0] / 2), -np.exp(1j * params[2]) * np.sin(params[0] / 2)],
                         [np.exp(1j * params[1]) * np.sin(params[0] / 2), np.exp(1j * (params[1] + params[2])) * np.cos(params[0] / 2)]], dtype=complex)
    elif ins_name in "p u1": # U1 gate
        return np.array([[1 + 0j, 0 + 0j],
                         [0 + 0j, np.exp(1j * params[0])]], dtype=complex)
    elif ins_name ==  "u2": # U1 gate
        return (1/np.sqrt(2)) * np.array([[1 + 0j, -np.exp(1j * params[1])],
                                          [np.exp(1j * params[0]), np.exp(1j * (params[0] + params[1]))]], dtype=complex)

# Returns probability of getting 0
def qu_prob(tgt_index, vector):
    res = 0.0

    for i in range(len(vector)):
        if ((i & (1 << tgt_index)) == 0):
            res += np.abs(vector[i]) ** 2

    return res

# Creates Measurement gate and applies it
def perform_measure(tgt_index):
    global quantum_circuit
    global random_states
    global measure_op_found
    global r_measure
    global statevector

    zero_gate = np.array([[1.0]], dtype=complex)
    one_gate = np.array([[1.0]], dtype=complex)

    zero_matrix = np.array([[1+0j, 0+0j],
                           [0+0j, 0+0j]], dtype=complex)
    one_matrix = np.array([[0+0j, 0+0j],
                          [0+0j, 1+0j]], dtype=complex)

    q_len = quantum_circuit.num_qubits

    # Creates measurement for 0 and 1 states
    for i in range(q_len):
        if i == (q_len - tgt_index - 1):
            zero_gate = np.kron(zero_gate, zero_matrix)
        else:
            zero_gate = np.kron(zero_gate, np.array([[1+0j, 0+0j],
                                                     [0+0j, 1+0j]], dtype=complex))
            
    for i in range(q_len):
        if i == (q_len - tgt_index - 1):
            one_gate = np.kron(one_gate, one_matrix)
        else:
            one_gate = np.kron(one_gate, np.array([[1+0j, 0+0j],
                                                   [0+0j, 1+0j]], dtype=complex))
            

    qubit_probability = 0 # Probability of the qubit being 0, 1 - qubit_probability is 1


    if (measure_op_found == False): # First time measurement
        for i in range(random_states): # Save measurement results for later use
            r = np.random.random()
            qubit_probability = qu_prob(tgt_index, statevector)
            if (r < qubit_probability):
                r_measure[i] = copy.deepcopy(np.dot(zero_gate, statevector))
            else:
                r_measure[i] = copy.deepcopy(np.dot(one_gate, statevector))
            
    else: # Second time measurement and beyond
        for i in range(random_states): # Save measurement results for later use
            r = np.random.random()
            qubit_probability = qu_prob(tgt_index, r_measure[i])
            if (r < qubit_probability):
                r_measure[i] = np.dot(zero_gate, r_measure[i])
            else:
                r_measure[i] = np.dot(one_gate, r_measure[i])
    

# Creates a gate that only applies to a single qubit
def create_gate_single(ins_name, qubit_tuple, tgt_index, params):
    global quantum_circuit
    gate = np.array([[1.0]], dtype=complex)

    matrix = grab_gate(ins_name, params)

    q_len = quantum_circuit.num_qubits

    for i in range(q_len):
        if i == (q_len - tgt_index - 1):
            gate = np.kron(gate, matrix)
        else:
            gate = np.kron(gate, np.array([[1+0j, 0+0j],
                                           [0+0j, 1+0j]], dtype=complex))

    return gate


# Entanglement operations on 2 qubits
def two_q_entanglement(ins_name, qubit_tuple, params, statevector):
    global quantum_circuit

    if ins_name in "cx cnot": # tuple[0] is the control qubit, tuple[1] is the target qubit
        ctrl_index = quantum_circuit.qubits.index(qubit_tuple[0])
        tgt_index = quantum_circuit.qubits.index(qubit_tuple[1])

        for i in range(len(statevector)):
            if ((i & (1 << ctrl_index)) != 0):
                j = i ^ (1 << tgt_index) # Flipped bit result

                if (i < j): # Prevent double flipping
                    statevector[i], statevector[j] = statevector[j], statevector[i] # Didn't even know python can do this


# Performs the instruction
def perform(ins, qubit_tuple):
    #QC.data.instruction.qubits[i]
    #QC.qubits.index(qubit) gives the index of the qubit obtained from the tuple qargs
    global quantum_circuit
    global r_measure
    #used for CNOT

    #Gate creation
    gate = None

    if (len(qubit_tuple) == 1):
        gate = create_gate_single(ins.operation.name, qubit_tuple, quantum_circuit.qubits.index(qubit_tuple[0]), ins.operation.params)
    elif (len(qubit_tuple) == 2):
        gate = 1
        pass

    return gate

# Grabs the QASM file as input from terminal
def fileselect():
    file = None

    while file == None: # Grabs File
        try:
            file = input("\nPath Address of QASM file?\nType without spaces: ")
        except FileNotFoundError:
            q_prompt = input("\n\nType \"Yes\" to exit: ")
            if q_prompt in "yes Yes":
                return None
            file = None

    return file

def warning_statement():
    print("\nMade by Tai Nguyen for FRI Spring 2025\n")
    print("Details:\n1. This is a light implementation of a Quantum Simulator")
    print("able to take in the typically used gates in class and CNOT.")
    print("This simulator can handle multi-qubit systems.")

    print("\n2. Does not allow any application of gates after measurement.")
    print("This simulator does not work with classical bits after measurement,")
    print("only using them when doing shots.")
    input("\nPress Enter to continue")
    return

# Does stuff with resulting statevector
def calculations_pt1_1(statevector):
    global quantum_circuit
    global deci

    two_power_q = 2 ** quantum_circuit.num_qubits
    # Errors
    if (statevector is None):
        print("QASM file is incompatible: there are issues within file or warnings regarding")
        print("the file that prevent this simulator from executing the file to completion.")
        print("Goodbye!")
        return None

    # testing
    # .real for real, .imag for complex
    # f format for binary: {<number>:0{<bit_amount>}b}

    deci = None

    try:
        deci = int(input("\nRound results to how many decimals?: "))
        if (deci < 0 or deci > 10):
            deci = 2
    except ValueError:
        print("Invalid input (number required)\nGoodbye!")
        return None


    # Factor calculations
    sq_sum = 0
    for i in range(two_power_q):
        sq_sum += np.abs(statevector[i]) ** 2

    normal_factor = np.round(np.sqrt(sq_sum), decimals=int(deci))


    # Phase is arctan of (b / a) where b is the complex part and a is real
    print("\nAmplitude for each state:\n-------------------------------------")
    for i in range(two_power_q):
        toprint = f"({i}) {i:0{quantum_circuit.num_qubits}b}: {np.round(statevector[i].real / normal_factor, decimals=int(deci))} + {np.round(statevector[i].imag / normal_factor, decimals=int(deci))}i"
        print(toprint)

    # If we wanted to know original amplitude and factor
    #print(f"\nNormalization factor (denominator): {normal_factor}")

    display_phase = input("\nType \"Yes\" to display phases: ")

    if (display_phase in "Yes yes"):
        print("\nPhases:\n-------------------------------------")
        for i in range(two_power_q):
            phase = np.arctan2(statevector[i].imag, statevector[i].real)
            toprint = f"({i}) {i:0{quantum_circuit.num_qubits}b}: {np.round(phase / np.pi, decimals=int(deci))}π"
            print(toprint)

# Allowed operations
def allowed_operations(ins):
    if (ins.operation.name in "id iden x y z h s sdg t tdg p u u1 u2 u3 cx cnot measure"):
        return True
    return False
    


# Does the calculations given by file, returns None if error, returns statevector otherwise
# Performs calculation of: Amplitude, Normalization factor, Phases
def calculations_pt1():
    global quantum_circuit
    global statevector
    global measure_op_found
    global random_states
    global qc_tuplelist
    global measure_sets
    global measure_last_op


    vis_num = -1
    
    while (vis_num < 0):
        try:
            vis_num = int(input(f"\nEnter visualization seed [0 - {random_states - 1}]: "))
            if (vis_num < 0 or vis_num > random_states - 1):
                raise ValueError("input out of range")
        except Exception:
            vis_num = input("Type \"Yes\" to exit: ")
            if (vis_num in "Yes yes"):
                return None
            vis_num = -1

    # Initialize gate
    gate = None

    # Create Statevector
    two_power_q = 2 ** quantum_circuit.num_qubits
    statevector = np.zeros(two_power_q, dtype=complex)
    statevector[0] = 1.0

    # Prevent multiple measurements of the same qubit
    measured = []

    # Performing each instruction
    for i in quantum_circuit.data:
        if (not allowed_operations(i)):
            print("Some operations not allowed in this simulator")
            return None
        
        if (i.operation.name != "barrier"):
            if (i.operation.name == "measure"): # Measurement operation case
                measure_last_op = True

                global altered_vector_state
                altered_vector_state.append(copy.deepcopy(statevector))

                m_idx = quantum_circuit.qubits.index(i.qubits[0])
                c_idx = quantum_circuit.clbits.index(i.clbits[0])
                if (m_idx in measured):
                    statevector = None
                    break

                try:
                    measured.append(m_idx)
                    qc_tuplelist.append((m_idx, c_idx, measure_sets)) # Tuple of qubit and classical bit
                    #raise Exception("Remeasurement error")
                except Exception:
                    print("\nMeasurement block resulted in error\nGoodbye!")
                

                perform_measure(quantum_circuit.qubits.index(i.qubits[0]))
                measure_op_found = True

            else: # Not a measurement operation
                if (measure_op_found): # No operations after measurements occur
                    print("Operations found after measurement, breaks simulator.")
                    return None
                
                if (measure_last_op == True):
                    measure_sets += 1
                    measure_last_op = False

                
                if (i.operation.name == "cx" or i.operation.name == "cnot"): # CNOT operation // Set up for other entanglement operations
                    if (measure_op_found):
                        for i in range(random_states):
                            two_q_entanglement(i.operation.name, i.qubits, i.operation.params, r_measure[i])

                    two_q_entanglement(i.operation.name, i.qubits, i.operation.params, statevector)

                else: # Non-CNOT operation
                    gate = perform(i, i.qubits)

                    if (gate is None):
                        print("Operations found in file not available or other errors.\nGoodbye!")
                        return None
                
                    #print(f"{gate}\n")
                    if (measure_op_found):
                        for i in range(random_states):
                            r_measure[i] = np.dot(gate, r_measure[i])

                    statevector = np.dot(gate, statevector)
            

        #print(f"{i.operation.name}")

    # Performs result calculations
    if (measure_op_found):
        calculations_pt1_1(r_measure[vis_num])
    else:
        calculations_pt1_1(statevector)

    return statevector
    
# Not in use
def prob_space_calc(): # Calculates the probability space for doing shots later
    global r_measure
    global random_states
    global measure_op_found
    global quantum_circuit
    global statevector

    total_out = 2 ** quantum_circuit.num_qubits

    prob_arr = []
    t_prob = 0.0

    if (measure_op_found):
        for m in range(random_states): # Different statevectors with randomized measured outcomes
            t_prob = 0.0
            SV_m = [] # Statevector for the mth measurement, contains probabilites of getting each state
            for state in range(total_out): # State is the outcome state i.e. 00, 01, 10, 11
                calc = (np.abs(r_measure[m][state]) ** 2)

                if (calc == 0.0):
                    SV_m.append(-1) # Placeholder for 0 for coding purposes
                else:
                    SV_m.append(calc + t_prob)
                    t_prob += calc
                # The plan is to traverse reversely and check if probability > random number

            prob_arr.append(copy.deepcopy(SV_m))
    else:
        for state in range(total_out): # State is the outcome state i.e. 00, 01, 10, 11
                calc = (np.abs(statevector[state]) ** 2)

                if (calc == 0.0):
                    prob_arr.append(-1)
                else:
                    prob_arr.append(calc + t_prob)
                    t_prob += calc
                # The plan is to traverse reversely and check if probability > random number

    return prob_arr

# Shot performing function
def perform_shots(shots):
    global r_measure
    global random_states
    global measure_op_found
    global quantum_circuit
    global statevector
    global qc_tuplelist
    global altered_vector_state
    
    total_out = 2 ** len(quantum_circuit.clbits)
    shots_arr = [0] * total_out

    for s in range(shots):
        val_idx = 0 # value serving as index

        for tup in range(len(qc_tuplelist)):
            if (measure_op_found): # Gets probability of 0
                prob_arr = qu_prob(qc_tuplelist[tup][0], altered_vector_state[qc_tuplelist[tup][2]])
            else:
                prob_arr = qu_prob(qc_tuplelist[tup][0], statevector)

            r = np.random.random()
            if (r >= prob_arr): # Translates to getting 1
                val_idx += 1 << qc_tuplelist[tup][1] # Adds the value of the qubit to the index

        shots_arr[val_idx] += 1 # Shot results in this state

    return shots_arr



def calculations_pt2(): # Performing shots
    global quantum_circuit
    global statevector
    global r_measure
    global measure_op_found
    global random_states
    global measure_sets
    global max_shots
    global deci


    # Grab amount of shots
    shots_num = -1
    while (shots_num < 0):
        try:
            shots_num = int(input(f"\nEnter Shots count (0 - {max_shots}]: "))
            if (shots_num <= 0 or shots_num > max_shots):
                raise ValueError("Input out of range")
        except ValueError:
            shots_num = input("Type \"Yes\" to exit: ")
            if (shots_num in "Yes yes"):
                return None
            
            shots_num = -1
            continue

    # Generate data for shots
    print(f"\nResults for {shots_num} shots\nAny 0 results are not printed.\n-------------------------------------")

    shots_arr = perform_shots(shots_num)
    shots_arr_prob = shots_arr.copy()
    for i in range(len(shots_arr_prob)):
        shots_arr_prob[i] /= shots_num

        if shots_arr[i] != 0:
            print(f"({i}) {i:0{quantum_circuit.num_qubits}b}: {shots_arr[i]}, %{np.round(shots_arr_prob[i] * 100, decimals=deci)}")
    

    return True


def main():
    global quantum_circuit
    global statevector
    global qc_tuplelist
    global r_measure
    global altered_vector_state
    
    # Warning messages
    warning_statement()

    file = fileselect() # Grabs file from user
    if (file == None): # For exiting
        return

    # Grab Circuit and draw
    print("\nDrawing of circuit from file:\n-------------------------------------")

    try:
        quantum_circuit = QuantumCircuit.from_qasm_file(file)
        print(quantum_circuit.draw(output='text'))
    except Exception:
        print("An error has occured, please retry")
        return


    retry = True # Meant for going through visualizations not recalculations

    statevector = calculations_pt1()



    sv = statevector.copy()
    rm = r_measure.copy()
    av = altered_vector_state.copy()
    bits = qc_tuplelist.copy()

    # Repeating options
    vis = True
    sho = True

    if (statevector is None): # Error message done in calculations
            return
    
    while (retry == True):
        
        if (sho == True):
            shot_option = input("\nType \"Yes\" to perform shots: ")
            if (shot_option in "Yes yes"):
                None_check = calculations_pt2()
                if (None_check is None):
                    return
                
            else:
                sho = False
            
        if (vis == True):
            if (input("\nType \"Yes\" for other visualizations: ") in "Yes yes"):
                reset_globals()
                statevector = sv
                r_measure = rm
                altered_vector_state = av
                qc_tuplelist = bits

                try:
                    vis_num = int(input(f"\nEnter visualization seed [0 - {random_states - 1}]: "))
                except Exception:
                    return
                
                calculations_pt1_1(r_measure[vis_num])
            else:
                vis = False

        if not (vis or sho):
            retry = False


main()
print("\nGoodbye!")
time.sleep(1.5)

