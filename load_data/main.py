import csv
import numpy as np


def load_csv(file_name):
    ant1, ant2, time = [], [], []
    u, v, vis = [], [], []
    sigma = []
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        row = next(reader)
        freq = float(row[2].split(':')[1][:-3])*1e9
        for row in reader:
            if row[0][0] == '#':
                continue
            time.append(float(row[0]))
            ant1.append(row[1])
            ant2.append(row[2])
            u.append(float(row[3]))
            v.append(float(row[4]))
            vis.append(float(row[5])*np.exp(1j*float(row[6])/180*np.pi))
            sigma.append(float(row[7]))
    return {
        'time': np.array(time),
        'ant1': np.array(ant1),
        'ant2': np.array(ant2),
        'u': np.array(u),
        'v': np.array(v),
        'vis': np.array(vis),
        'sigma': np.array(sigma),
        'freq': freq,
    }


print(load_csv("1.csv"))
