import numpy as np
import csv
import matplotlib.pyplot as plt
import skrf as rf
from skrf import Network, Frequency
from numpy.linalg import inv
from tqdm import tqdm
import torch

#This function leaves only the predefined ports and renumbering the ports

#This function generates a PDN with n rows and m columns
def pdn_predefined(n,m, uc_all, map, PDN_name):    
    num_uc=len(uc_all)
    s5p=[]
    for i in range(num_uc):
        if uc_all[i].shape[1]==5:
            s5p.append(i+1)
        
    new_map=[]
    total_port=0

    for k in tqdm(range(n)):
        port_num=0
        print(map[k,:])
        for i in tqdm(range(m)):
            num=int(map[k][i])
            #once for the first uc of a row
            if i%m==0:
                if num in s5p:
                    merged_z=uc_all[num-1]
                    new_map.append(num)
                    current_port=1
                    port_num+=1
                
                else:
                    merged_z=uc_all[num-1]
                    current_port=0
                    port_num=0
                
               
            #uc to be merged
            else:   
                if num in s5p:
                    uc=uc_all[num-1]
                    new_map.append(num)
                    current_port=1
  
                else:
                    uc=uc_all[num-1]
                    current_port=0

             
                i=i-1  
                seg1=merged_z[:,0:i+2,0:i+2]  
                seg2=merged_z[:,0:i+2,i+2:i+3]
                seg3=merged_z[:,0:i+2,i+3:] 
                seg4=merged_z[:,i+2:i+3,0:i+2] 
                seg5=merged_z[:,i+2:i+3,i+2:i+3]
                seg6=merged_z[:,i+2:i+3,i+3:]
                seg7=merged_z[:,i+3:,0:i+2] 
                seg8=merged_z[:,i+3:,i+2:i+3]
                seg9=merged_z[:,i+3:,i+3:]
                
                ## define each z components  
                ## axis = 2 (row-wise), axis = 1 (column-wise)
                seg_temp1=np.concatenate([np.array(seg1), np.array(seg3)], axis=2)
                seg_temp2=np.concatenate([np.array(seg7), np.array(seg9)], axis=2)
            
                zaa=np.concatenate([np.array(seg_temp1), np.array(seg_temp2)], axis=1)
                zbb=uc[:,1:,1:]
                zpp=seg5
                zqq=uc[:,0:1,0:1]
                zap=np.concatenate([np.array(seg2), np.array(seg8)], axis=1)
                zpa=np.concatenate([np.array(seg4), np.array(seg6)], axis=2)
                zqb=uc[:,0:1,1:]
                zbq=uc[:,1:,0:1]
  
                
                
                ## computation of merged z 
                Z_AA_temp=np.matmul(zap, inv(zpp+zqq))
                Z_AA=zaa-np.matmul(Z_AA_temp, zpa)
                
                Z_AB_temp=np.matmul(zap, inv(zpp+zqq))
                Z_AB=np.matmul(Z_AB_temp, zqb)
                
                Z_BB_temp=np.matmul(zbq, inv(zpp+zqq))
                Z_BB=zbb-(np.matmul(Z_BB_temp, zqb))
            
                Z_BA_temp=np.matmul(zbq, inv(zpp+zqq))
                Z_BA=np.matmul(Z_BA_temp, zpa)
                
                Z_temp1=np.concatenate([np.array(Z_AA), np.array(Z_AB)], axis=2)
                Z_temp2=np.concatenate([np.array(Z_BA), np.array(Z_BB)], axis=2)
                Z_temp=np.concatenate([np.array(Z_temp1), np.array(Z_temp2)], axis=1)
               

                a=i+2
                b=2*i+3+port_num
                c=2*i+6+port_num
                d=2*i+7+port_num
                    
                m1=Z_temp[:, 0:a, 0:a]
                m2=Z_temp[:, 0:a, a:b]
                m3=Z_temp[:, 0:a, b:c]
                m4=Z_temp[:, 0:a, c:d]
                m5=Z_temp[:, a:b, 0:a]
                m6=Z_temp[:, a:b, a:b]
                m7=Z_temp[:, a:b, b:c]
                m8=Z_temp[:, a:b, c:d]
                m9=Z_temp[:, b:c, 0:a]
                m10=Z_temp[:, b:c, a:b]
                m11=Z_temp[:, b:c, b:c]
                m12=Z_temp[:, b:c, c:d]
                m13=Z_temp[:, c:d, 0:a]
                m14=Z_temp[:, c:d, a:b]
                m15=Z_temp[:, c:d, b:c]
                m16=Z_temp[:, c:d, c:d]
            
                ## Re-arrange  Port numbering 
                if current_port==1:
                    temp1=np.concatenate([np.array(m1), np.array(m3), np.array(m2), np.array(m4)], axis=2)
                    temp2=np.concatenate([np.array(m9), np.array(m11), np.array(m10), np.array(m12)], axis=2)
                    temp3=np.concatenate([np.array(m5), np.array(m7), np.array(m6), np.array(m8)], axis=2)
                    temp4=np.concatenate([np.array(m13), np.array(m15), np.array(m14), np.array(m16)], axis=2)
                    merged_z=np.concatenate([np.array(temp1), np.array(temp2), np.array(temp3), np.array(temp4)], axis=1)
                    port_num=port_num+1
                
                elif current_port==0:
                    temp1=np.concatenate([np.array(m1), np.array(m3), np.array(m2)], axis=2)
                    temp2=np.concatenate([np.array(m9), np.array(m11), np.array(m10)], axis=2)
                    temp3=np.concatenate([np.array(m5), np.array(m7), np.array(m6)], axis=2)
                    merged_z=np.concatenate([np.array(temp1), np.array(temp2), np.array(temp3)], axis=1)

        if k==0:
            Z_up=merged_z
            total_port=port_num
            seg1=merged_z[:, 1:1+m, 1:1+m]
            seg2=merged_z[:, 2+2*m:, 2+2*m:]
            seg3=merged_z[:, 1:1+m, 2+2*m:]
            seg4=merged_z[:, 2+2*m:, 1:1+m]
            temp1=np.concatenate([np.array(seg1), np.array(seg3)], axis=2)
            temp2=np.concatenate([np.array(seg4), np.array(seg2)], axis=2)
            Z_up=np.concatenate([np.array(temp1), np.array(temp2)], axis=1)
            
        elif k!=0:
            Z_lo=merged_z

            ##upper block Z-segmentation            
            zpp=Z_up[:, :m, :m]
            zaa=Z_up[:, m:m+total_port, m:m+total_port]
            zpa=Z_up[:, :m, m:m+total_port]
            zap=Z_up[:, m:m+total_port, :m]
            
            ## lower block Z-segmentation
            lo_1=Z_lo[:, 1:1+m, 1:1+m]
            lo_2=Z_lo[:, 1:1+m, 2+m:2+2*m]
            lo_3=Z_lo[:, 1:1+m, 2+2*m:2+2*m+port_num]
            lo_4=Z_lo[:, 2+m:2+2*m, 1:1+m]
            lo_5=Z_lo[:, 2+m:2+2*m, 2+m:2+2*m]
            lo_6=Z_lo[:, 2+m:2+2*m, 2+2*m:2+2*m+port_num]
            lo_7=Z_lo[:, 2+2*m:2+2*m+port_num, 1:1+m]
            lo_8=Z_lo[:, 2+2*m:2+2*m+port_num, 2+m:2+2*m]
            lo_9=Z_lo[:, 2+2*m:2+2*m+port_num, 2+2*m:2+2*m+port_num]
            
            zbb_temp1=np.concatenate([np.array(lo_1), np.array(lo_3)], axis=2)
            zbb_temp2=np.concatenate([np.array(lo_7), np.array(lo_9)], axis=2)
            zbb=np.concatenate([np.array(zbb_temp1), np.array(zbb_temp2)], axis=1)
            zqq=lo_5
            zbq=np.concatenate([np.array(lo_2), np.array(lo_8)], axis=1)
            zqb=np.concatenate([np.array(lo_4), np.array(lo_6)], axis=2)
              
            ## computation of merged z    
            Z_AA_temp=np.matmul(zap, inv(zpp+zqq))
            Z_AA=zaa-np.matmul(Z_AA_temp, zpa)
            Z_AB_temp=np.matmul(zap, inv(zpp+zqq))
            Z_AB=np.matmul(Z_AB_temp, zqb)
            Z_BB_temp=np.matmul(zbq, inv(zpp+zqq))
            Z_BB=zbb-(np.matmul(Z_BB_temp, zqb))
            Z_BA_temp=np.matmul(zbq, inv(zpp+zqq))
            Z_BA=np.matmul(Z_BA_temp, zpa)
            Z_temp1=np.concatenate([np.array(Z_AA), np.array(Z_AB)], axis=2)
            Z_temp2=np.concatenate([np.array(Z_BA), np.array(Z_BB)], axis=2)
            Z_temp=np.concatenate([np.array(Z_temp1), np.array(Z_temp2)], axis=1)
            #rearrange in order of m for connection and rest for the ports
            
            
            #total port: ports on Z_up
            #port_num: ports on
            Z_1=Z_temp[:, :total_port, :total_port]
            Z_2=Z_temp[:, :total_port, total_port:total_port+m]
            Z_3=Z_temp[:, :total_port, total_port+m:total_port+m+port_num]
            Z_4=Z_temp[:, total_port:total_port+m, :total_port]
            Z_5=Z_temp[:, total_port:total_port+m, total_port:total_port+m]
            Z_6=Z_temp[:, total_port:total_port+m, total_port+m:total_port+m+port_num]
            Z_7=Z_temp[:, total_port+m:total_port+m+port_num, :total_port]
            Z_8=Z_temp[:, total_port+m:total_port+m+port_num, total_port:total_port+m]
            Z_9=Z_temp[:, total_port+m:total_port+m+port_num, total_port+m:total_port+m+port_num]
            Ztemp1=np.concatenate([np.array(Z_5), np.array(Z_4), np.array(Z_6)], axis=2)
            Ztemp2=np.concatenate([np.array(Z_2), np.array(Z_1), np.array(Z_3)], axis=2)
            Ztemp3=np.concatenate([np.array(Z_8), np.array(Z_7), np.array(Z_9)], axis=2)
            Z=np.concatenate([np.array(Ztemp1), np.array(Ztemp2), np.array(Ztemp3)], axis=1)
            
            Z_up=Z
            total_port=total_port+port_num
            
        #print(np.shape(Z_up))
        #print("total_port_num", total_port)

        
    num_port=np.size(Z_up,1)-total_port

    #remove side ports
    PDN=Z_up[:,num_port:,num_port:]
    print("PDN shape", np.shape(PDN))
    print("new map", new_map)

    return(PDN, new_map)