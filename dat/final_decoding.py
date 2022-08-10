import numpy as np
import matplotlib.pyplot as plt
import ldpc
import bposd
import csv

def insert(list1, list2, index):
    try:
        list1[index] = list2
    except:
        while len(list1) < index+1:
            list1.append([])
        list1[index] = list2

def read_code_info(qubitsX, qubitsZ, syndromesX, syndromesZ):
    with open("data.csv", newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            p = float(line[0])
            data = [int(i) for i in line[4:]]
            if line[2] == 'q':
                if line[3] == 'X':
                    which = qubitsX
                else:
                    which = qubitsZ
            else:
                if line[3] == 'X':
                    which = syndromesX
                else:
                    which = syndromesZ
            try:
                insert(which[p],data,int(line[1]))
            except:
                which[p] = [];
                insert(which[p],data,int(line[1]))

def read_matrix(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter=' ')
        matrix = []
        for line in reader:
            matrix.append([int(i) for i in line])
    return matrix

def count_failures(qubitsX, qubitsZ, syndromesX, syndromesZ, hx, hz, lx, lz):
    ps = [i for i in qubitsX]
    ps.sort()
    qubitsX = np.array([qubitsX[i] for i in ps])
    qubitsZ = np.array([qubitsZ[i] for i in ps])
    syndromesX = np.array([syndromesX[i] for i in ps])
    syndromesZ = np.array([syndromesZ[i] for i in ps])
    failuresX = np.zeros((len(ps),len(lx)))
    failuresZ = np.zeros((len(ps),len(lx)))
    runs = np.zeros(len(ps))
    for i in range(len(ps)):
        runs[i] = len(qubitsX[i])
        bpd_x=ldpc.bposd_decoder(
                hx,
                error_rate=ps[i],
                channel_probs=[None],
                max_iter=416,
                bp_method="ms",
                ms_scaling_factor=0,
                osd_method="osd_cs",
                osd_order=7
                )
        bpd_z=ldpc.bposd_decoder(
                hz,
                error_rate=ps[i],
                channel_probs=[None],
                max_iter=416,
                bp_method="ms",
                ms_scaling_factor=0,
                osd_method="osd_cs",
                osd_order=7
                )
        for j in range(len(qubitsX[i])):
            bpd_x.decode(syndromesX[i][j])
            qubitsZ[i][j] = (qubitsZ[i][j] + bpd_x.osdw_decoding) % 2
            failuresZ[i] += lx@qubitsZ[i][j]%2
            bpd_z.decode(syndromesZ[i][j])
            qubitsX[i][j] = (qubitsX[i][j] + bpd_z.osdw_decoding) % 2
            failuresX[i] += lz@qubitsX[i][j]%2
    return ps, failuresX, failuresZ, runs

def write_params(ps, runs):
    with open("params.csv", 'w') as f:
        f.write(str(ps[0]))
        for i in range(len(ps)-1):
            f.write(',' + str(ps[i+1]))
        f.write('\n')
        f.write(str(runs[0]))
        for i in range(len(runs)-1):
            f.write(',' + str(runs[i+1]))

def write_fails(pauli, array):
    with open(pauli+"_fails.csv", 'w') as f:
        for row in array:
            f.write(str(row[0]))
            for i in range(len(row)-1):
                f.write(',' + str(row[i+1]))
            f.write('\n')

qubitsX = {}
qubitsZ = {}
syndromesX = {}
syndromesZ = {}
read_code_info(qubitsX,qubitsZ,syndromesX,syndromesZ)
hx = np.array(read_matrix("parity_check_matrices/lifted_product_[[416,18,20]]_hx.txt"))
hz = np.array(read_matrix("parity_check_matrices/lifted_product_[[416,18,20]]_hz.txt"))
lx = np.array(read_matrix("parity_check_matrices/lifted_product_[[416,18,20]]_lx.txt"))
lz = np.array(read_matrix("parity_check_matrices/lifted_product_[[416,18,20]]_lz.txt"))
ps, failuresX, failuresZ, runs = count_failures(qubitsX, qubitsZ, syndromesX, syndromesZ, hx, hz, lx, lz)
write_params(ps, runs)
write_fails("X", failuresX)
write_fails("Z", failuresZ)
