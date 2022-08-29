import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit

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

def x(L):
    return lambda p, pth, v : (p-pth)*(L**(1/v))

def ansatz(x,a0,a1,a2):
    return a0 + a1*x + a2*(x**2)

def func(p,pth,v,a0,a1,a2):
    perL = len(p)//4
    xs = [x(L) for L in [14,18,24,32]]
    xdata = []
    for i in range(len(p)):
        xdata.append(xs[i//perL](p[i],pth,v))
    xdata = np.array(xdata)
    return ansatz(xdata,a0,a1,a2)

#process data
data14 = np.array(consolidate_data(read_data_file("2022-08-29_5-2/data14.csv")))
data18 = np.array(consolidate_data(read_data_file("2022-08-29_5-2/data18.csv")))
data24 = np.array(consolidate_data(read_data_file("2022-08-29_5-2/data24.csv")))
data32 = np.array(consolidate_data(read_data_file("2022-08-29_5-2/data32.csv")))

#normal plotting
plt.errorbar(data14[0], data14[1], yerr=data14[2], label="L=14", linestyle='', marker='o', color='blue')
plt.errorbar(data18[0], data18[1], yerr=data18[2], label="L=18", linestyle='', marker='^', color='orange')
plt.errorbar(data24[0], data24[1], yerr=data24[2], label="L=24", linestyle='', marker='v', color='green')
plt.errorbar(data32[0], data32[1], yerr=data32[2], label="L=32", linestyle='', marker='s', color='red')
plt.semilogy()
plt.legend()
plt.show()

#fitting to ansatz
data = np.hstack((data14,data18,data24,data32))
popt, pcov = curve_fit(func, data[0], data[1], sigma=data[2], p0=[0.036,1,1,1,1])

plt.plot(x(14)(data14[0], popt[0], popt[1]),data14[1], label="L=14", linestyle='', marker='o')
plt.plot(x(18)(data18[0], popt[0], popt[1]),data18[1], label="L=18", linestyle='', marker='^')
plt.plot(x(24)(data24[0], popt[0], popt[1]),data24[1], label="L=24", linestyle='', marker='v')
plt.plot(x(32)(data32[0], popt[0], popt[1]),data32[1], label="L=32", linestyle='', marker='s')
plt.show()

print(popt[0])
print(np.sqrt(pcov[0][0]))
