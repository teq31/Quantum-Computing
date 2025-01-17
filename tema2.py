import numpy as np 
from numpy import arcsin, sqrt, pi
from numpy import linalg as nl
from qiskit import *
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Operator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')

print("Exercitiul 1\n")
print("Circuitul pentru prima stare\n")

circuit=QuantumCircuit(2)
circuit.h(0)
circuit.x(1)
circuit.s(1)
circuit.cx(0,1)
circuit.h(1)
print(circuit)

print("Circuitul pentru a doua stare\n")

theta=2*arcsin(sqrt(2/3))
circuit2=QuantumCircuit(3)
circuit2.ry(theta,0)
circuit2.cx(0,1)
circuit2.cx(0,2)
print(circuit2)

print("Exercitiul 2\n")
print("a)\n")

circuit3=QuantumCircuit(1)
circuit3.h(0)
circuit3.p(pi/2,0)
print(circuit3)

print("b)\n")

bell=QuantumCircuit(2)
bell.h(0)
bell.cx(0,1)
print(bell)

print("c)\n")

circuit4=QuantumCircuit(3,2)

circuit4.h(0)
circuit4.p(pi/2,0)

circuit4.h(1)
circuit4.cx(1,2)

circuit4.cx(0,1)
circuit4.h(0)
circuit4.measure([0,1], [0,1])

circuit4.cx(1,2)
circuit4.cz(0,2)

print(circuit4)

backend = AerSimulator()
circuit_compilat = transpile(circuit4, backend)
job_sim = backend.run(circuit_compilat, shots=1024)

rezultat_simulare = job_sim.result()
cnt = rezultat_simulare.get_counts(circuit_compilat)

print("Rezultatele masuratorii: ", cnt)
plot_histogram(cnt, color="pink")
plt.show()

print("Exercitiul 3\n")

theta = np.exp(2j*np.pi/3)

def starea1(circuit):
    circuit.x(2)
    circuit.h(1)
    circuit.cp(2*np.angle(theta**2),1,2)
    circuit.cx(1,0)
    circuit.p(np.angle(theta),2)

def starea2(circuit):
    circuit.x(0)
    circuit.h(1)
    circuit.cp(2*np.angle(theta),1,0)
    circuit.cx(1,2)
    circuit.p(np.angle(theta**2),2)

def distinge_stari(circuit):
    circuit.barrier()
    circuit.h(0)
    circuit.crz(np.pi/3, 0, 1)
    circuit.h(1)
    circuit.barrier()
    circuit.measure([0,1,2], [0,1,2])

def masuram(circuit):
    backend = AerSimulator()
    job = backend.run(transpile(circuit, backend), shots=1024)
    cnt = job.result().get_counts()

    cnt_phi0 = cnt.get('000', 0)
    cnt_phi1 = cnt.get('111', 0)

    if cnt_phi0 > cnt_phi1:
        return 0
    else:
        return 1

print("Starea |φ₀⟩:\n")

circuit_phi0=QuantumCircuit(3,3)
starea1(circuit_phi0)
print(circuit_phi0)

print("Starea |φ₁⟩:\n")

circuit_phi1=QuantumCircuit(3,3)
starea2(circuit_phi1)
print(circuit_phi1)

distinge_stari(circuit_phi0)
print("Noua stare |φ₀⟩:\n")
print(circuit_phi0)

distinge_stari(circuit_phi1)
print("Noua stare |φ₁⟩:\n")
print(circuit_phi1)

cnt = masuram(circuit_phi0)
print(cnt)