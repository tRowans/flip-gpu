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

def read_code_info(filename, qubitsX, qubitsZ, syndromesX, syndromesZ):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            p = int(line[0])
            data = [int(i) for i in line[4:]]
            if line[2] == 'q':
                if line[3] == 'X':
                    which = qubitsX
                else:
                    which = qubitsZ
            else:
                if line[3] == 'X':
                    which = syndromeX
                else:
                    which = syndromeZ
            try:
                insert(which[p],data,int(line[1]))
            except:
                which[p] = [];
                insert(which[p],data,int(line[1]))

def read_matrix(filename)
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        matrix = []
        for line in reader:
            matrix.append([int(i) for i in line])

def count_failures(qubitsX, qubitsZ, syndromesX, syndromesZ, hx, hz, lx, lz):
    ps = [i for i in qubitsX]
    ps.sort()
    qubitsX = np.array([qubitsX[i] for i in ps])
    qubitsZ = np.array([qubitsZ[i] for i in ps])
    syndromeX = np.array([syndromeX[i] for i in ps])
    syndromeZ = np.array([syndromeZ[i] for i in ps])
    failuresX = np.zeros((len(ps),18))
    failuresZ = np.zeros((len(ps),18))
    runs = np.zeros(len(ps))
    for i in range(len(ps)):
        runs[i] = len(qubitsX[i])
        bpd_x=bposd_decoder(
                hx,
                error_rate=p,
                channel_probs=[None],
                max_iter=416,
                bp_method="ms",
                ms_scaling_factor=0,
                osd_method="osd_cs",
                osd_order=7
                )
        bpd_z=bposd_decoder(
                hz,
                error_rate=p,
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

qubitsX = {}
qubitsZ = {}
syndromeX = {}
syndromeZ = {}
read_code_info("data.csv",qubitsX,qubitsZ,syndromeX,syndromeZ)
hx = np.array(read_matrix("parity_check_matrices/lifted_product_[[416,18,20]]_hx.txt"))
hz = np.array(read_matrix("parity_check_matrices/lifted_product_[[416,18,20]]_hz.txt"))
lx = np.array(read_matrix("parity_check_matrices/lifted_product_[[416,18,20]]_lx.txt"))
lz = np.array(read_matrix("parity_check_matrices/lifted_product_[[416,18,20]]_lz.txt"))
ps, failuresX, failuresZ, runs = check_failures(qubitsX, qubitsZ, syndromesX, syndromesZ, hx, hz, lx, lz)


