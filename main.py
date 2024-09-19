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

PDN_name='gaudi'  # 저장할 파일 구분을 위한 이름

#PDN size 정의
int_N=1000 #세로 unit cell 개수
int_M=1000  #가로 unit cell 개수

# 활용하고자 하는 모든 unit cell snp 파일 불러오기 및 numpy로 변환
snp1=rf.Network('data/S_para/240919_11um_UC.s5p')
uc1=rf.network.s2z(snp1._s, z0=50)

"""
snp1=rf.Network('data/S_para/ex1.s4p')
uc1=rf.network.s2z(snp1._s, z0=50)

snp2=rf.Network('data/S_para/ex6.s5p')
uc2=rf.network.s2z(snp2._s, z0=50)

snp3=rf.Network('data/S_para/ex3.s4p')
uc3=rf.network.s2z(snp3._s, z0=50)

snp4=rf.Network('data/S_para/ex4.s4p')
uc4=rf.network.s2z(snp4._s, z0=50)

snp5=rf.Network('data/S_para/ex5.s4p')
uc5=rf.network.s2z(snp5._s, z0=50)
"""
# 모든 uc를 하나의 list로 묶기
#uc=[uc1, uc2, uc3, uc4, uc5]
uc=[uc1]
# 각 unit cell로 구성된 PDN의 map을 저장한 numpy 파일 불러오기.
# map은 unit cell의 개수와 동일한 크기의 numpy array로 uc1의 위치에 1, uc2의 위치에 2, ... uc5의 위치에 5로 표시.
# 예) [1,1,2,3,4] 가로 5개, 세로 1개의 PDN의 map. 각 숫자에 해당하는 unit cell이 위치한 곳에 숫자 표시.

PDN_map=np.ones((int_N, int_M))
#PDN_map=np.array([[3,1,2,2,2,2], [3,1,2,2,2,2], [3,1,2,2,2,2]])

#print(PDN_map)

#################################################################################
#################################################################################
#################################################################################


# PDN 생성 코드 실행
PDN, new_map=PDN_gen.pdn_predefined(int_N, int_M, uc, PDN_map, PDN_name)

# 생성된 PDN 저장 
np.save('final_model/%d_by_%d_PDN_%s.npy'%(int_N, int_M,PDN_name), PDN)

# 생성된 PDN의 포트 정보 맵 저장
np.save('final_model/final_map_%s.npy'%PDN_name, new_map)


# plot을 위한 주파수 포인트 저장 
freq=snp1.f
freq=np.array(freq)

#read csv file
ver=pd.read_csv('verification.csv', sep=',',header=0)
ver=np.array(ver)

#print(ver[:5,1])
print(np.abs(PDN[:5,0,0]))
print(ver.shape)
#plot
plt.figure(1)
plt.plot(freq, np.abs(PDN[:,0,0]), label='port 1', color='red')
plt.plot(freq, ver[:,1], label='ads', color='blue')
#set log scale
plt.xscale('log')
plt.yscale('log')

plt.savefig('final_model/%d_by_%d_PDN_%s.png'%(int_N, int_M,PDN_name))
