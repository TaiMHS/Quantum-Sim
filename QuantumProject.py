import qiskit, os, pathlib
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import *

# This Python File is created by Tai Nguyen
# Latest edit date: 4/13/2025

# Globals (potentially used for searching specific states)
quantum_circuit: QuantumCircuit = None

# To-do
# Make a method that performs the right gates to the right qubit (done, used kronecker products)
# Make a method that computes the state before measurement (done, statevector made)
# make a counts input and use it to create shots
# Perform proper state vector calcs

# Other stuff
# Make a text display of results |More to do|

# Implement measure operation:
# sum amplitude squared where measured qubit is 1 or 0 (group the probabilities according to the qubit value)
# choose between 1 or 0 based on probability, 0 out other amplitudes
# zero out other amplitudes


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


# Performs the instruction
def perform(ins, qubit_tuple):
    #QC.data.instruction.qubits[i]
    #QC.qubits.index(qubit) gives the index of the qubit obtained from the tuple qargs
    global quantum_circuit
    #used for CNOT

    #Gate creation
    gate = None

    if not (ins.operation.name in "id iden x y z h s sdg t tdg p u u1 u2 u3 cx cnot"):
        return None

    if (len(qubit_tuple) == 1):
        gate = create_gate_single(ins.operation.name, qubit_tuple, quantum_circuit.qubits.index(qubit_tuple[0]), ins.operation.params)
    elif (len(qubit_tuple) == 2):
        #gate = create_gate_double()
        pass

    return gate

# Grabs the QASM file as input from terminal
def fileselect():
    dir = None

    while dir == None: # Grabs directory
        try:
            dirname = input("\nDirectory of QASM file?\nType without spaces: ")
            dir = os.listdir(dirname)
        except FileNotFoundError:
            q_prompt = input("\n\nType \"Yes\" to exit: ")
            if q_prompt in "yes Yes":
                return None
            dir = None
    
    print("\n-------------------------------------")
    for f in dir:
        if f.endswith(".qasm"):
            print(f + "\n") 

    filename = None

    while filename == None: # Grabs file name
        filename = input("\nType a file name from above or \"Exit\" to exit: ")

        if (filename in "Exit exit"):
            return None

        if not (".qasm" in filename) or not (filename in dir):
            filename = None


    
    return os.path.join(dirname, filename)

def warning_statement():
    print("\nMade by Tai Nguyen for FRI Spring 2025\n")
    print("Details:\n1. This is a light implementation of a Quantum Simulator")
    print("able to take in the typically used gates in class and CNOT.")
    print("This simulator can handle multi-qubit systems.")

    print("\n2. Does not allow any application of gates after measurement.")
    print("This simulator does not work with classical bits after measurement or classical bits in general.")
    input("\nPress Enter to continue")
    return


# Does the calculations given by file, returns None if error, returns statevector otherwise
# Performs calculation of: Amplitude, Normalization factor, Phases
def calculations_pt1():
    global quantum_circuit
    # Create Statevector
    two_power_q = 2 ** quantum_circuit.num_qubits
    statevector = np.zeros(two_power_q, dtype=complex)
    statevector[0] = 1.0

    # Prevent multiple measurements of the same qubit
    measured = []

    # Performing each instruction
    for i in quantum_circuit.data:
        if (i.operation.name != "barrier"):
            if (i.operation.name == "measure"):
                m_idx = quantum_circuit.qubits.index(i.qubits[0])
                if (m_idx in measured):
                    statevector = None
                    break

                try:
                    measured.append()
                except Exception:
                    print("\nMeasurement block resulted in error\nGoodbye!")


            gate = perform(i, i.qubits)


            if (gate is None):
                print("Operations found in file not available or other errors.\nGoodbye!")
                return None
            
            #print(f"{gate}\n")
            statevector = np.dot(gate, statevector)
            

        #print(f"{i.operation.name}")

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
        deci = int(input("Round results to how many decimals?: "))
    except ValueError:
        print("Invalid input (number required)\nGoodbye!")
        return None


    # Phase is arctan of (b / a) where b is the complex part and a is real
    print("\nAmplitude for each state:\n-------------------------------------")
    for i in range(two_power_q):
        toprint = f"({i}) {i:0{quantum_circuit.num_qubits}b}: {np.round(statevector[i].real, decimals=int(deci))} + {np.round(statevector[i].imag, decimals=int(deci))}i"
        print(toprint)

    # Factor calculations
    sq_sum = 0
    for i in range(two_power_q):
        sq_sum += np.abs(statevector[i]) ** 2

    normal_factor = np.round(np.sqrt(sq_sum), decimals=int(deci))

    print(f"\n Normalization factor (denominator): {normal_factor}")

    display_phase = input("\nType \"Yes\" to display phases: ")

    if (display_phase in "Yes yes"):
        print("\nPhases:\n-------------------------------------")
        for i in range(two_power_q):
            phase = np.arctan2(statevector[i].imag, statevector[i].real)
            toprint = f"({i}) {i:0{quantum_circuit.num_qubits}b}: {np.round(phase / np.pi, decimals=int(deci))}π"
            print(toprint)

    return statevector

def main():
    global quantum_circuit
    
    # Warning messages
    warning_statement()

    file = fileselect()
    if (file == None): # For exiting
        return
    #file = 'C:\\' + 'Users\Tie\Downloads\G.qasm' # testing

    # Grab Circuit and draw
    print("\nDrawing of circuit from file:\n-------------------------------------")

    try:
        quantum_circuit = QuantumCircuit.from_qasm_file(file)
        print(quantum_circuit.draw(output='text'))
    except Exception:
        print("An error has occured, please retry")


    statevector = calculations_pt1()
    if (statevector is None): # Error message done in calculations
        return

    
    


    

main()
print("\nGoodbye!")
