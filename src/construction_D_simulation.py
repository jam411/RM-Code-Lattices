### Construction D simulation
# 1. Prepare trainied decoder model each codes.
# 2. Loading Model
#   2-1. Set decoder paramters
#   2-2. Generate Codes
#   2-3. Generate Decoders
#   2-4. Load Trained Model to Decoders
#   2-5. Set generator matrices.
# 3. Set Simulation Parameters
#   3-1. n, num_levels, k
#   3-2. vnr ranges
#   3-3. runs and err
# 4. Consider decoders inputs in decoding.


# Import modules
import sys
import os

from learned_BP import *  # noqa
from itertools import product  # noqa
from tqdm import tqdm as tqdm

from 'utils/utils' import *

# Set CUDA parameter
CUDA = False

# Set BP decoder parameter
model_kwargs = {
    'T': 5,
    'tie': True,
    'use_cuda': CUDA,
    'mode': 'plain',
    'damping_init': 1,
    'damping_train': False,
}

# Create Code
codeOne = RM_Code(32, 6, mode='oc', use_cuda=CUDA)
codeTwo = RM_Code(32, 26, mode='oc', use_cuda=CUDA)

# Create Decoder Network
modelOne = BP_Decoder(codeOne, **model_kwargs, **extra_model)
modelTwo = BP_Decoder(codeTwo, **model_kwargs, **extra_model)

# Load trained network
modelOne.load_state_dict(torch.load("./model_RM_32_6sysBP.pt"))
modelTwo.load_state_dict(torch.load("./model_RM_32_26sysreduceBP.pt"))

G1,G2 = codeOne.returnG(), codeTwo.returnG()
generator_matrices = [G1, G2]

# Construction D Simulation Parameters
n = 32
num_level = 2
k = [6,26,n]

vnr_min = 5.0
vnr_max = 5.0
vnr_step = 0.5

max_runs = 200000
max_err = 1000

# Set construction D Params
vnr_vec = []
while True:
    vnr_vec.append(vnr_min)
    if vnr_min == vnr_max: break
    vnr_min += vnr_step

num_block_err = [0 for _ in range(len(vnr_vec))]
num_symbol_err = [0 for _ in range(len(vnr_vec))]
num_run = [0 for _ in range(len(vnr_vec))]

info_bits = [[0] for _ in range(num_level+1)]
est_info = [[0] for _ in range(num_level+1)]
coded = [[0] for _ in range(num_level+1)]
decoded = [[0] for _ in range(num_level+1)]
est_coded = [[0] for _ in range(num_level+1)]

for i in range(num_level+1):
    info_bits[i] = [0 for _ in range(k[i])]
    est_info[i] = [0 for _ in range(k[i])]
    coded[i] = [0 for _ in range(n)]
    decoded[i] = [0 for _ in range(n)]
    est_coded[i] = [0 for _ in range(n)]

k_total = 0
for i in range(num_level): k_total += k[i]
volume = np.power(2.0, num_level*n-k_total)
norm_volume = np.power(volume, (2.0/n))

# Main loop
for i_vnr in range(0,len(vnr_vec)):
    vnr_linear = np.power(10.0, vnr_vec[i_vnr] / 10.0)
    sigma2 = norm_volume / (2.0 * np.pi * np.exp(1.0) * vnr_linear)
    print(sigma2)

    for run in tqdm(range(max_runs)):
        if num_block_err[i_vnr] > max_err: continue

        # Generate random bits
        for i in range(num_level):
            for j in range(k[i]):
                info_bits[i][j] = np.random.randint(0,2)
        
        # Encode (over real)
        for i in range(num_level): coded[i] = encode(generator_matrices[i], info_bits[i])
        coded[num_level] = np.zeros(n) # lift-up
        
        # Construction D
        x = GenLattice(coded)

        # Channel
        y = AWGN(x, sigma2)
        
        # Multi-stage decoding
        for level in range(num_level):
            
            # Culculate LLR
            llr = (1 - (2*mod_triangle(y))) / (2*sigma2)
            
            # Decoders
            param = torch.Tensor([i_vnr for _ in range(1)]) # for AdaBP
            llrs = torch.Tensor([[llr[i] for _ in range(1)] for i in range(n)])
                        
            ##### Notice: Decoders inputs #####
            if level == 0:
                for modelLen in range(len([i[-1].tolist() for i in modelOne(llrs)[-1]])):
                    decoded[level][modelLen] = hard_decision([i[-1].tolist() for i in modelOne(llrs)[-1]][modelLen])
            else:
                for modelLen in range(len([i[-1].tolist() for i in modelTwo(llrs)[-1]])):
                    decoded[level][modelLen] = hard_decision([i[-1].tolist() for i in modelTwo(llrs)[-1]][modelLen])
#             # AdaBP
#             if level == 0:
#                 for modelLen in range(len([i[-1].tolist() for i in modelOne((llrs,param))[-1]])):
#                     decoded[level][modelLen] = hard_decision([i[-1].tolist() for i in modelOne((llrs,param))[-1]][modelLen])
#             else:
#                 for modelLen in range(len([i[-1].tolist() for i in modelTwo((llrs,param))[-1]])):
#                     decoded[level][modelLen] = hard_decision([i[-1].tolist() for i in modelTwo((llrs,param))[-1]][modelLen])
#             ##### ##### ##### ##### ##### #####

            # Retrieve information (first k bits)
            for i in range(k[level]): est_info[level][i] = decoded[level][i]
            
            # Re-encode
            est_coded[level] = encode(generator_matrices[level], est_info[level])
            
            # Subtract & scaling
            y = Subtract(y, est_coded[level]) 

        # Round to the nearest integer
        for i in range(n): est_coded[num_level][i] = round(y[i])
        
        # Estimated lattice points
        est_x = [0 for _ in range(n)]
        for i in range(n):
            for j in range(num_level+1):
                est_x[i] += (1 << j) * est_coded[j][i]
        
        # Count errors
        symbol_error = 0
        for i in range(n):
            if x[i] != est_x[i]: symbol_error += 1
        
        num_run[i_vnr] += 1
        num_symbol_err[i_vnr] += (symbol_error/n)
        if symbol_error: num_block_err[i_vnr] += 1

#         if run % 10000 == 0:
#             print("VNR:", vnr_vec[i_vnr]," TRIAL:",num_run[i_vnr]," ERROR:",num_block_err[i_vnr]," WER:",num_block_err[i_vnr] / num_run[i_vnr]," SER:",num_symbol_err[i_vnr] / num_run[i_vnr])
        # run loop end

    print("VNR:", vnr_vec[i_vnr]," WER:",num_block_err[i_vnr] / num_run[i_vnr]," SER:",num_symbol_err[i_vnr] / num_run[i_vnr])

    if num_block_err[i_vnr] == 0:
        for i in range(i_vnr+1,len(vnr_vec)):
            num_block_err[i] = 0
            num_run[i] = 1
        break
    # main loop end