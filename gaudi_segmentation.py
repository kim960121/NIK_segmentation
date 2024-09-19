import numpy as np
import skrf as rf
from skrf import Network, Frequency
import torch
import habana_frameworks.torch.core as htcore
from tqdm import tqdm

def process_chunk(chunk, device):
    return torch.tensor(chunk, dtype=torch.float32).to(device)

def compute_merged_z(zap, zpa, zpp, zqq, zbq, zqb, zaa, zbb, device):
    with torch.no_grad():
        Z_AA_temp = torch.matmul(zap, torch.inverse(zpp + zqq))
        Z_AA = zaa - torch.matmul(Z_AA_temp, zpa)
        Z_AB = torch.matmul(Z_AA_temp, zqb)
        Z_BB_temp = torch.matmul(zbq, torch.inverse(zpp + zqq))
        Z_BB = zbb - torch.matmul(Z_BB_temp, zqb)
        Z_BA = torch.matmul(Z_BB_temp, zpa)
    return Z_AA, Z_AB, Z_BB, Z_BA

def pdn_predefined(n, m, uc_all, map, PDN_name, chunk_size=100):
    htcore.hpu_initialize()
    num_gpus = torch.hpu.device_count()
    print(f"Using {num_gpus} Gaudi GPUs")

    devices = [torch.device(f"hpu:{i}") for i in range(num_gpus)]
    current_device = 0

    num_uc = len(uc_all)
    s5p = [i + 1 for i, uc in enumerate(uc_all) if uc.shape[1] == 5]
    new_map = []
    total_port = 0

    uc_all_processed = []
    for uc in tqdm(uc_all, desc="Processing unit cells"):
        chunks = [uc[i:i+chunk_size] for i in range(0, uc.shape[0], chunk_size)]
        processed_chunks = [process_chunk(chunk, devices[i % num_gpus]) for i, chunk in enumerate(chunks)]
        uc_all_processed.append(torch.cat(processed_chunks, dim=0))

    for k in tqdm(range(n), desc="Processing rows"):
        port_num = 0
        #print(map[k, :])
        for i in tqdm(range(m), desc="Processing columns"):
            num = int(map[k][i])
            if i % m == 0:
                if num in s5p:
                    merged_z = uc_all_processed[num - 1]
                    new_map.append(num)
                    current_port = 1
                    port_num += 1
                else:
                    merged_z = uc_all_processed[num - 1]
                    current_port = 0
                    port_num = 0
            else:
                if num in s5p:
                    uc = uc_all_processed[num - 1]
                    new_map.append(num)
                    current_port = 1
                else:
                    uc = uc_all_processed[num - 1]
                    current_port = 0

                i = i - 1
                seg1, seg2, seg3 = merged_z[:, :i+2, :i+2], merged_z[:, :i+2, i+2:i+3], merged_z[:, :i+2, i+3:]
                seg4, seg5, seg6 = merged_z[:, i+2:i+3, :i+2], merged_z[:, i+2:i+3, i+2:i+3], merged_z[:, i+2:i+3, i+3:]
                seg7, seg8, seg9 = merged_z[:, i+3:, :i+2], merged_z[:, i+3:, i+2:i+3], merged_z[:, i+3:, i+3:]

                zaa = torch.cat([torch.cat([seg1, seg3], dim=2), torch.cat([seg7, seg9], dim=2)], dim=1)
                zbb = uc[:, 1:, 1:]
                zpp = seg5
                zqq = uc[:, 0:1, 0:1]
                zap = torch.cat([seg2, seg8], dim=1)
                zpa = torch.cat([seg4, seg6], dim=2)
                zqb = uc[:, 0:1, 1:]
                zbq = uc[:, 1:, 0:1]

                # Distribute computation across GPUs
                Z_AA, Z_AB, Z_BB, Z_BA = [], [], [], []
                for start in range(0, zaa.shape[0], chunk_size):
                    end = min(start + chunk_size, zaa.shape[0])
                    device = devices[current_device]
                    current_device = (current_device + 1) % num_gpus

                    z_aa, z_ab, z_bb, z_ba = compute_merged_z(
                        zap[start:end].to(device), zpa[start:end].to(device),
                        zpp[start:end].to(device), zqq[start:end].to(device),
                        zbq[start:end].to(device), zqb[start:end].to(device),
                        zaa[start:end].to(device), zbb[start:end].to(device),
                        device
                    )
                    Z_AA.append(z_aa.cpu())
                    Z_AB.append(z_ab.cpu())
                    Z_BB.append(z_bb.cpu())
                    Z_BA.append(z_ba.cpu())

                Z_AA, Z_AB, Z_BB, Z_BA = [torch.cat(z, dim=0) for z in (Z_AA, Z_AB, Z_BB, Z_BA)]
                Z_temp = torch.cat([torch.cat([Z_AA, Z_AB], dim=2), torch.cat([Z_BA, Z_BB], dim=2)], dim=1)

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

            # Distribute computation across GPUs
            Z_AA, Z_AB, Z_BB, Z_BA = [], [], [], []
            for start in range(0, zaa.shape[0], chunk_size):
                end = min(start + chunk_size, zaa.shape[0])
                device = devices[current_device]
                current_device = (current_device + 1) % num_gpus

                z_aa, z_ab, z_bb, z_ba = compute_merged_z(
                    zap[start:end].to(device), zpa[start:end].to(device),
                    zpp[start:end].to(device), zqq[start:end].to(device),
                    zbq[start:end].to(device), zqb[start:end].to(device),
                    zaa[start:end].to(device), zbb[start:end].to(device),
                    device
                )
                Z_AA.append(z_aa.cpu())
                Z_AB.append(z_ab.cpu())
                Z_BB.append(z_bb.cpu())
                Z_BA.append(z_ba.cpu())

            Z_AA, Z_AB, Z_BB, Z_BA = [torch.cat(z, dim=0) for z in (Z_AA, Z_AB, Z_BB, Z_BA)]
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

    return PDN.numpy(), new_map

if __name__ == "__main__":
    PDN_name = 'gaudi'
    int_N, int_M = 1000, 1000

    snp1 = rf.Network('data/S_para/240919_11um_UC.s5p')
    uc1 = rf.network.s2z(snp1._s, z0=50)

    uc = [uc1]
    PDN_map = np.ones((int_N, int_M))

    PDN, new_map = pdn_predefined(int_N, int_M, uc, PDN_map, PDN_name)

    np.save(f'final_model/{int_N}_by_{int_M}_PDN_{PDN_name}.npy', PDN)
    np.save(f'final_model/final_map_{PDN_name}.npy', new_map)

htcore.hpu_finalize()