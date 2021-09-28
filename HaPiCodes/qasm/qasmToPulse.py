from qutip.qip.qasm import read_qasm
qc = read_qasm("example.qasm")
for g in qc.gates:
    print(g)