import numpy as np
from numpy import linalg as nl
from qiskit import *
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.quantum_info import Operator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')

def matricea_adjuncta(A):
    A_adjuncta = A.transpose()
    for i in range (len(A_adjuncta)):
        for j in range (len(A_adjuncta)):
            A_adjuncta[i][j]=A_adjuncta[i][j].conj()
    return(A_adjuncta) 

def produs_tensorial(a,b):
    matrice_ab=np.array([[a[0][0]*b[0][0]], 
                 [a[0][0]*b[1][0]],
                 [a[1][0]*b[0][0]],
                 [a[1][0]*b[1][0]]], dtype = complex)
    return matrice_ab


print("Exercitiul 1 \n")

def exercitiul1(a,b,c):
    matrice_ab=produs_tensorial(a,b)
    matrice_ca=produs_tensorial(c,a)
    matrice_ca_adjuncta=matricea_adjuncta(matrice_ca)
    rezultat=np.matmul(matrice_ab,matrice_ca_adjuncta)
    return rezultat

matrice_a=np.array([[1], [1]], dtype = complex)

matrice_b=np.array([[2], [-2]], dtype = complex)

matrice_c=np.array([[4], [-1]], dtype = complex)

print("Rezultatul |ab><ca| este \n", exercitiul1(matrice_a,matrice_b,matrice_c))


print("Exercitiul 2\n")


A = (1/np.sqrt(2)) * np.array([[0, 1], [1, 0]], dtype=complex)

def exercitiul2(A):
    n = A.shape[0]
    I = np.zeros((n, n), dtype=complex) 
    for i in range(n):
        I[i][i] = 1
   
    A_adjuncta=matricea_adjuncta(A)
   
    x=np.matmul(A,A_adjuncta)
    y=np.matmul(A_adjuncta,A)

    epsilon=1e-6
    if nl.norm(np.subtract(x,I)) < epsilon and nl.norm(np.subtract(y,I)) < epsilon:
        print("Matricea este unitara\n")
    else:
        print("Matricea nu este unitara\n")

exercitiul2(A)

print("Exercitiul 3\n")
print("a)")
circuit=QuantumCircuit(2)
circuit.h(0)
circuit.h(1)
circuit.cx(0,1)
circuit.h(0)
circuit.h(1)
circuit.swap(0,1)
print(circuit)

print("b)")
U=Operator(circuit)
U.data
print("Matricea U:\n")
print(U.data)
exercitiul2(U.data)

print("c)")
circuit_c=QuantumCircuit(2,2)
circuit_c.h(0)
circuit_c.h(1)
circuit_c.cx(0,1)
circuit_c.h(0)
circuit_c.h(1)
circuit_c.swap(0,1)
circuit_c.h(0)
print(circuit_c)
circuit_c.measure(0,0)
circuit_c.measure(1,1)
print(circuit_c)

backend = AerSimulator()
qc_comp= transpile(circuit_c, backend)
job_sim = backend.run(qc_comp, shots=2000)

rezultat_simulare = job_sim.result()
cnt = rezultat_simulare.get_counts(qc_comp)
print(cnt)

plot_histogram(cnt, color="blue")
plt.show()

print("d)")

def StareaBell00():
    circuit_d=QuantumCircuit(2,2)
    circuit_d.h(0)
    circuit_d.cx(0,1)
    circuit_d.barrier()

    circuit_d.h(0)
    circuit_d.h(1)
    circuit_d.cx(0,1)
    circuit_d.h(0)
    circuit_d.h(1)
    circuit_d.swap(0,1)
    circuit_d.h(0)
    print(circuit_d)
    circuit_d.measure(0,0)
    circuit_d.measure(1,1)
    print(circuit_d)

    backend = AerSimulator()
    qc_comp= transpile(circuit_d, backend)
    job_sim = backend.run(qc_comp, shots=2000)

    rezultat_simulare = job_sim.result()
    cnt = rezultat_simulare.get_counts(qc_comp)
    print(cnt)

    plot_histogram(cnt, color="blue")
    plt.show()
print("Initiem cu starea Bell 00")
StareaBell00()

def StareaBell01():
    circuit_d=QuantumCircuit(2,2)
    circuit_d.h(0)
    circuit_d.x(1)
    circuit_d.cx(0,1)
    circuit_d.barrier()

    circuit_d.h(0)
    circuit_d.h(1)
    circuit_d.cx(0,1)
    circuit_d.h(0)
    circuit_d.h(1)
    circuit_d.swap(0,1)
    circuit_d.h(0)
    print(circuit_d)
    circuit_d.measure(0,0)
    circuit_d.measure(1,1)
    print(circuit_d)

    backend = AerSimulator()
    qc_comp= transpile(circuit_d, backend)
    job_sim = backend.run(qc_comp, shots=2000)

    rezultat_simulare = job_sim.result()
    cnt = rezultat_simulare.get_counts(qc_comp)
    print(cnt)

    plot_histogram(cnt, color="blue")
    plt.show()
print("Initiem cu starea Bell 01")
StareaBell01()

def StareaBell10():
    circuit_d=QuantumCircuit(2,2)
    circuit_d.x(0)
    circuit_d.h(0)
    circuit_d.cx(0,1)
    circuit_d.barrier()

    circuit_d.h(0)
    circuit_d.h(1)
    circuit_d.cx(0,1)
    circuit_d.h(0)
    circuit_d.h(1)
    circuit_d.swap(0,1)
    circuit_d.h(0)
    print(circuit_d)
    circuit_d.measure(0,0)
    circuit_d.measure(1,1)
    print(circuit_d)

    backend = AerSimulator()
    qc_comp= transpile(circuit_d, backend)
    job_sim = backend.run(qc_comp, shots=2000)

    result_simulator = job_sim.result()
    cnt = result_simulator.get_counts(qc_comp)
    print(cnt)

    plot_histogram(cnt, color="blue")
    plt.show()
print("Initiem cu starea Bell 10")
StareaBell10()

def StareaBell11():
    circuit_d=QuantumCircuit(2,2)
    circuit_d.x(0)
    circuit_d.x(1)
    circuit_d.h(0)
    circuit_d.cx(0,1)
    circuit_d.barrier()

    circuit_d.h(0)
    circuit_d.h(1)
    circuit_d.cx(0,1)
    circuit_d.h(0)
    circuit_d.h(1)
    circuit_d.swap(0,1)
    circuit_d.h(0)
    print(circuit_d)
    circuit_d.measure(0,0)
    circuit_d.measure(1,1)
    print(circuit_d)

    backend = AerSimulator()
    qc_comp= transpile(circuit_d, backend)
    job_sim = backend.run(qc_comp, shots=2000)

    rezultat_simulare = job_sim.result()
    cnt = rezultat_simulare.get_counts(qc_comp)
    print(cnt)

    plot_histogram(cnt, color="blue")
    plt.show()
print("Initiem cu starea Bell 11")
StareaBell11()

print("Exercitiul 4")

vector1=(1/np.sqrt(2))*np.array([[1], [0], [0], [1]], dtype = complex)
                  
vector2=1/2*np.array([[1], [1], [1], [1]], dtype = complex)

def entangled(vector): 
    count_0 = 0
    for i in range(4):
        if vector[i][0] == 0:
            count_0 = count_0+1

    if count_0 == 0:
        if (vector[0][0]==vector[2][0]) or (vector[1][0]==vector[3][0]):
            return False
        else: 
            return True
    elif count_0 == 1:
        return True
    elif count_0 == 2:
        if (vector[0][0]== 0 and vector[3][0] == 0) or (vector[1][0]== 0 and vector[2][0] == 0):
            return True
        else:
            return False
    elif count_0 == 3 or count_0==4:
        return False
    
print(entangled(vector1))
print(entangled(vector2))
