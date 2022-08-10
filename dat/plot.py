import numpy as np
import matplotlib.pyplot as plt
import ldpc
import bposd
import csv

def read_params():
    with open("params.csv", newline='') as f:
        reader = csv.reader(f)
        params = [[float(i) for i in line] for line in reader]
        ps = params[0]
        runs = [int(i) for i in params[1]]
    return ps, runs

def read_fails(pauli):
    with open(pauli+"_fails.csv", newline='') as f:
        reader = csv.reader(f)
        fails = [[int(float(i)) for i in line] for line in reader]
    return fails;

def agresti_coull(f,n):
    f = f + 2
    n = n + 4
    p_fail = f/n
    error_bar = 2*np.sqrt((p_fail*(1-p_fail))/n)
    return p_fail, error_bar

def to_probs(X_fails, Z_fails, runs):
    pLx = np.zeros((len(X_fails), len(X_fails[0])))
    pLz = np.zeros((len(Z_fails), len(Z_fails[0])))
    barX = np.zeros((len(X_fails),len(X_fails[0])))
    barZ = np.zeros((len(Z_fails),len(Z_fails[0])))
    for i in range(len(X_fails)):
        for j in range(len(X_fails[i])):
            pLx[i][j], barX[i][j] = agresti_coull(X_fails[i][j], runs[i])
            pLz[i][j], barZ[i][j] = agresti_coull(Z_fails[i][j], runs[i])
    return pLx, barX, pLz, barZ

def plotting_separate(name, ps, pLs, bar):
    fig, axs = plt.subplots(3,6)
    fig.suptitle(name)
    for i in range(18):
        axs[i//6,i%6].errorbar(ps,pLs[:,i],bar[:,i], marker='o', linestyle='')

def plotting_average(name, ps, pLs, bar):
    av_pLs = [sum(i)/len(i) for i in pLs]
    av_bar = [np.sqrt(sum(i**2))/len(i) for i in bar]
    plt.figure()
    plt.title(name)
    plt.errorbar(ps, av_pLs, av_bar, marker='o', linestyle='')

ps, runs = read_params()
X_fails = read_fails('X')
Z_fails = read_fails('Z')
pLx, barX, pLz, barZ = to_probs(X_fails, Z_fails, runs)

plotting_average('X', ps, pLx, barX)
#plt.semilogy()
plt.savefig("X.eps", format='eps')
plotting_average('Z', ps, pLz, barZ)
#plt.semilogy()
plt.savefig("Z.eps", format='eps')

