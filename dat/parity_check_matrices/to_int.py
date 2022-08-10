import csv
import sys

matrix = []
with open(sys.argv[1], newline='') as f:
    reader = csv.reader(f, delimiter=' ')
    for line in reader:
        matrix.append([int(float(i)) for i in line])
with open(sys.argv[1], 'w') as f:
    for row in matrix:
        f.write(str(row[0]))
        for i in range(len(row)-1):
            f.write(' ' + str(row[i+1]))
        f.write('\n')
