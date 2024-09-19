import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import skrf as rf
from skrf import Network, Frequency
import torch
from tqdm import tqdm

def pdn_predefined(n, m, uc_all, map, PDN_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_uc = len(uc_all)
    s5p = [i + 1 for i, uc in enumerate(uc_all) if uc.shape[1] == 5]
    new_map = []
    total_port = 0

    # Convert uc_all to PyTorch tensors and move to GPU
    uc_all = [torch.tensor(uc, dtype=torch.complex64).to(device) for uc in uc_all]

    for k in tqdm(range(n)):
        port_num = 0
        print(map[k, :])
        for i in tqdm(range(m)):
            num = int(map[k][i])
            if i % m == 0:
                if num in s5p:
                    merged_z = uc_all[num - 1]
                    new_map.append(num)
                    current_port = 1
                    port_num += 1
                else:
                    merged_z = uc_all[num - 1]
                    current_port = 0
                    port_num = 0
            else:
                if num in s5p:
                    uc = uc_all[num - 1]
                    new_map.append(num)
                    current_port = 1
                else:
                    uc = uc_all[num - 1]
                    current_port = 0

                i = i - 1
                # Segment the merged_z tensor
                seg1, seg2, seg3 = merged_z[:, :i+2, :i+2], merged_z[:, :i+2, i+2:i+3], merged_z[:, :i+2, i+3:]
                seg4, seg5, seg6 = merged_z[:, i+2:i+3, :i+2], merged_z[:, i+2:i+3, i+2:i+3], merged_z[:, i+2:i+3, i+3:]
                seg7, seg8, seg9 = merged_z[:, i+3:, :i+2], merged_z[:, i+3:, i+2:i+3], merged_z[:, i+3:, i+3:]

                # Concatenate segments
                zaa = torch.cat([torch.cat([seg1, seg3], dim=2), torch.cat([seg7, seg9], dim=2)], dim=1)
                zbb = uc[:, 1:, 1:]
                zpp = seg5
                zqq = uc[:, 0:1, 0:1]
                zap = torch.cat([seg2, seg8], dim=1)
                zpa = torch.cat([seg4, seg6], dim=2)
                zqb = uc[:, 0:1, 1:]
                zbq = uc[:, 1:, 0:1]

                # Compute merged z
                Z_AA_temp = torch.matmul(zap, torch.inverse(zpp + zqq))
                Z_AA = zaa - torch.matmul(Z_AA_temp, zpa)
                Z_AB = torch.matmul(Z_AA_temp, zqb)
                Z_BB_temp = torch.matmul(zbq, torch.inverse(zpp + zqq))
                Z_BB = zbb - torch.matmul(Z_BB_temp, zqb)
                Z_BA = torch.matmul(Z_BB_temp, zpa)

                Z_temp = torch.cat([torch.cat([Z_AA, Z_AB], dim=2), torch.cat([Z_BA, Z_BB], dim=2)], dim=1)

                # Re-arrange Port numbering
                a, b, c, d = i+2, 2*i+3+port_num, 2*i+6+port_num, 2*i+7+port_num
                m1, m2, m3, m4 = Z_temp[:, :a, :a], Z_temp[:, :a, a:b], Z_temp[:, :a, b:c], Z_temp[:, :a, c:d]
                m5, m6, m7, m8 = Z_temp[:, a:b, :a], Z_temp[:, a:b, a:b], Z_temp[:, a:b, b:c], Z_temp[:, a:b, c:d]
                m9, m10, m11, m12 = Z_temp[:, b:c, :a], Z_temp[:, b:c, a:b], Z_temp[:, b:c, b:c], Z_temp[:, b:c, c:d]
                m13, m14, m15, m16 = Z_temp[:, c:d, :a], Z_temp[:, c:d, a:b], Z_temp[:, c:d, b:c], Z_temp[:, c:d, c:d]

                if current_port == 1:
                    temp1 = torch.cat([m1, m3, m2, m4], dim=2)
                    temp2 = torch.cat([m9, m11, m10, m12], dim=2)
                    temp3 = torch.cat([m5, m7, m6, m8], dim=2)
                    temp4 = torch.cat([m13, m15, m14, m16], dim=2)
                    merged_z = torch.cat([temp1, temp2, temp3, temp4], dim=1)
                    port_num += 1
                elif current_port == 0:
                    temp1 = torch.cat([m1, m3, m2], dim=2)
                    temp2 = torch.cat([m9, m11, m10], dim=2)
                    temp3 = torch.cat([m5, m7, m6], dim=2)
                    merged_z = torch.cat([temp1, temp2, temp3], dim=1)

        if k == 0:
            Z_up = merged_z
            total_port = port_num
            seg1, seg2 = Z_up[:, 1:1+m, 1:1+m], Z_up[:, 2+2*m:, 2+2*m:]
            seg3, seg4 = Z_up[:, 1:1+m, 2+2*m:], Z_up[:, 2+2*m:, 1:1+m]
            Z_up = torch.cat([torch.cat([seg1, seg3], dim=2), torch.cat([seg4, seg2], dim=2)], dim=1)
        else:
            Z_lo = merged_z

            zpp = Z_up[:, :m, :m]
            zaa = Z_up[:, m:m+total_port, m:m+total_port]
            zpa = Z_up[:, :m, m:m+total_port]
            zap = Z_up[:, m:m+total_port, :m]

            lo_1, lo_2, lo_3 = Z_lo[:, 1:1+m, 1:1+m], Z_lo[:, 1:1+m, 2+m:2+2*m], Z_lo[:, 1:1+m, 2+2*m:2+2*m+port_num]
            lo_4, lo_5, lo_6 = Z_lo[:, 2+m:2+2*m, 1:1+m], Z_lo[:, 2+m:2+2*m, 2+m:2+2*m], Z_lo[:, 2+m:2+2*m, 2+2*m:2+2*m+port_num]
            lo_7, lo_8, lo_9 = Z_lo[:, 2+2*m:2+2*m+port_num, 1:1+m], Z_lo[:, 2+2*m:2+2*m+port_num, 2+m:2+2*m], Z_lo[:, 2+2*m:2+2*m+port_num, 2+2*m:2+2*m+port_num]

            zbb = torch.cat([torch.cat([lo_1, lo_3], dim=2), torch.cat([lo_7, lo_9], dim=2)], dim=1)
            zqq = lo_5
            zbq = torch.cat([lo_2, lo_8], dim=1)
            zqb = torch.cat([lo_4, lo_6], dim=2)

            Z_AA_temp = torch.matmul(zap, torch.inverse(zpp + zqq))
            Z_AA = zaa - torch.matmul(Z_AA_temp, zpa)
            Z_AB = torch.matmul(Z_AA_temp, zqb)
            Z_BB_temp = torch.matmul(zbq, torch.inverse(zpp + zqq))
            Z_BB = zbb - torch.matmul(Z_BB_temp, zqb)
            Z_BA = torch.matmul(Z_BB_temp, zpa)
            Z_temp = torch.cat([torch.cat([Z_AA, Z_AB], dim=2), torch.cat([Z_BA, Z_BB], dim=2)], dim=1)

            Z_1, Z_2, Z_3 = Z_temp[:, :total_port, :total_port], Z_temp[:, :total_port, total_port:total_port+m], Z_temp[:, :total_port, total_port+m:total_port+m+port_num]
            Z_4, Z_5, Z_6 = Z_temp[:, total_port:total_port+m, :total_port], Z_temp[:, total_port:total_port+m, total_port:total_port+m], Z_temp[:, total_port:total_port+m, total_port+m:total_port+m+port_num]
            Z_7, Z_8, Z_9 = Z_temp[:, total_port+m:total_port+m+port_num, :total_port], Z_temp[:, total_port+m:total_port+m+port_num, total_port:total_port+m], Z_temp[:, total_port+m:total_port+m+port_num, total_port+m:total_port+m+port_num]

            Z = torch.cat([torch.cat([Z_5, Z_4, Z_6], dim=2), torch.cat([Z_2, Z_1, Z_3], dim=2), torch.cat([Z_8, Z_7, Z_9], dim=2)], dim=1)

            Z_up = Z
            total_port = total_port + port_num

    num_port = Z_up.size(1) - total_port
    PDN = Z_up[:, num_port:, num_port:]
    print("PDN shape", PDN.shape)
    print("new map", new_map)

    return PDN.cpu().numpy(), new_map

