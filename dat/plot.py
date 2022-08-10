import numpy as np
import matplotlib.pyplot as plt
import csv

def read_data_file(filename):
    data = [[],[],[]]
    with open("{}".format(filename), newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            data[0].append(float(line[1]))  #p
            data[1].append(int(line[3]))  #number of runs
            data[2].append(int(line[-1]))   #number of logical errors
    return data

def agresti_coull(f,n):
    f = f + 2
    n = n + 4
    p_fail = f/n
    error_bar = 2*np.sqrt((p_fail*(1-p_fail))/n)
    return p_fail, error_bar

def consolidate_data(data):
    xvals = {i for i in data[0]}
    newdata = [[],[],[]]
    for i in xvals:
        newdata[0].append(i)
        n_runs = 0
        n_fails = 0
        for j in range(len(data[0])):
            if data[0][j] == i:
                n_runs += data[1][j]
                n_fails += data[2][j]
        p_fail, error_bar = agresti_coull(n_fails,n_runs)
        newdata[1].append(p_fail)
        newdata[2].append(error_bar)
    return newdata

data14 = consolidate_data(read_data_file("2022-07-01/data14.csv"))
data18 = consolidate_data(read_data_file("2022-07-01/data18.csv"))
data24 = consolidate_data(read_data_file("2022-07-01/data24.csv"))
data32 = consolidate_data(read_data_file("2022-07-01/data32.csv"))

plt.errorbar(data14[0], data14[1], yerr=data14[2], label="L=14", linestyle='', marker='o')
plt.errorbar(data18[0], data18[1], yerr=data18[2], label="L=18", linestyle='', marker='^')
plt.errorbar(data24[0], data24[1], yerr=data24[2], label="L=24", linestyle='', marker='v')
plt.errorbar(data32[0], data32[1], yerr=data32[2], label="L=32", linestyle='', marker='s')
plt.semilogy()
plt.legend()
plt.show()
