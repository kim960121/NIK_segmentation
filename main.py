import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import skrf as rf
from skrf import Network, Frequency
from numpy.linalg import inv
import PDN_gen
np.set_printoptions(threshold=np.inf)


#################################################################################
#################################### INPUTS #####################################
#################################################################################

# Main execution

PDN_name = 'gaudi'
int_N, int_M = 1000, 1000

snp1 = rf.Network('data/S_para/240919_11um_UC.s5p')
uc1 = rf.network.s2z(snp1._s, z0=50)

uc = [uc1]
PDN_map = np.ones((int_N, int_M))

PDN, new_map = PDN_gen.pdn_predefined(int_N, int_M, uc, PDN_map, PDN_name)

np.save(f'final_model/{int_N}_by_{int_M}_PDN_{PDN_name}.npy', PDN)
np.save(f'final_model/final_map_{PDN_name}.npy', new_map)

freq = np.array(snp1.f)
"""
ver = pd.read_csv('verification.csv', sep=',', header=0).values

plt.figure(1)
plt.plot(freq, np.abs(PDN[:, 0, 0]), label='port 1', color='red')
plt.plot(freq, ver[:, 1], label='ads', color='blue')
plt.xscale('log')
plt.yscale('log')
plt.savefig(f'final_model/{int_N}_by_{int_M}_PDN_{PDN_name}.png')
"""