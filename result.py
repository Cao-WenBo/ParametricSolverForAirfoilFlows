import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import pickle
from scipy.spatial import KDTree
import os 
import shutil
from ParametricSolver import Solver, Net

from matplotlib import rcParams,colors
my_font1 = {"family":"Times New Roman", "size":22, "style":"italic"}

globalConfig = {
    "font.family":'serif',
    "font.size": 20,
    "font.weight": 'normal',
    "font.serif": ['Times New Roman'],
    "mathtext.fontset":'stix',
}
rcParams.update(globalConfig)
# %% 
device = torch.device("cuda:0" if 1 else "cpu")

data = np.fromfile('mesh/NACA0012_CMesh_400_150.x', sep=' ')
I = int(data[0]); J = int(data[1])
xxyy = data[2:].reshape(2*J, I)
xx = xxyy[:J,:].T; yy = xxyy[J:,:].T

model = torch.load('model/Model 1.pth')
nn = Solver(device, model, xx, yy, use_cst=True)  ### use_cst=False for Model 2

# %%
from case_params import case_params_list

plt.figure(figsize=(15,8))
error = []
for i in range(6):
    case_params = case_params_list[i]
    Re0 = torch.tensor(case_params['Re']).reshape(1,-1) / nn.Re_factor
    alpha0 = torch.tensor(case_params['alpha']/180*torch.pi).reshape(1,-1)
    b_up0 = torch.tensor(case_params['cst_up']).reshape(1,-1)
    b_down0 = torch.tensor(case_params['cst_down']).reshape(1,-1)
    state_para0 = torch.cat([Re0, alpha0, b_up0, b_down0], dim=1)
    
    nn.Predict_state_para(state_para0)
    Cp = nn.Cp.cpu().detach().numpy()
    Cf_f = nn.Cf_f.cpu().detach().numpy()
    
    Data = np.loadtxt(case_params_list[i]['ref_Cp_path'], skiprows=1)[:,1:]
    kd_tree = KDTree(Data[:,:2])
    distances, indices = kd_tree.query(nn.wall['X'].cpu().numpy(), k=2)
    Data_sorted = (Data[indices[:,0]] * distances[:,1:2] + Data[indices[:,1]] * distances[:,0:1]) / (distances.sum(1, keepdims=True))
    Cp_ref = Data_sorted[:,2:3]
    Cf_ref = Data_sorted[:,3:4]
    
    x0  = nn.wall['x0'].cpu().detach().numpy()
    ind = x0 < 0.49
    
    error.append(np.abs(Cp[ind]-Cp_ref[ind]).sum()/np.abs(Cp_ref[ind]).sum())

    plt.subplot(2,3,i+1)
    plt.plot(x0, Cp_ref, lw=3, label='Ref')
    plt.plot(x0, Cp, '--', lw=3, label='Pred')
    
    plt.gca().invert_yaxis()
    plt.xlabel('$x$')
    plt.ylabel('$C_p$')
    plt.title('Case '+str(i+1))
    plt.legend()
plt.tight_layout()

print(np.array(error).mean())

plt.figure(figsize=(15,8))
for i in range(6):
    case_params = case_params_list[i]
    Re0 = torch.tensor(case_params['Re']).reshape(1,-1) / nn.Re_factor
    alpha0 = torch.tensor(case_params['alpha']/180*torch.pi).reshape(1,-1)
    b_up0 = torch.tensor(case_params['cst_up']).reshape(1,-1)
    b_down0 = torch.tensor(case_params['cst_down']).reshape(1,-1)
    state_para0 = torch.cat([Re0, alpha0, b_up0, b_down0], dim=1)
    
    nn.Predict_state_para(state_para0)
    Cp = nn.Cp.cpu().detach().numpy()
    Cf_f = nn.Cf_f.cpu().detach().numpy()
    
    Data = np.loadtxt(case_params_list[i]['ref_Cp_path'], skiprows=1)[:,1:]
    kd_tree = KDTree(Data[:,:2])
    distances, indices = kd_tree.query(nn.wall['X'].cpu().numpy(), k=2)
    Data_sorted = (Data[indices[:,0]] * distances[:,1:2] + Data[indices[:,1]] * distances[:,0:1]) / (distances.sum(1, keepdims=True))
    Cp_ref = Data_sorted[:,2:3]
    Cf_ref = Data_sorted[:,3:4]
    
    x0 = nn.wall['x0'].cpu().detach().numpy()
    x0_f = nn.wall['X_f'][:,0].cpu().detach().numpy()

    plt.subplot(2,3,i+1)
    plt.plot(x0, Cf_ref, lw=3, label='Ref')
    x0_f = np.r_[x0_f[-1:],x0_f]
    Cf_f = np.r_[Cf_f[-1:],Cf_f]
    plt.plot(x0_f, np.abs(Cf_f), '--', lw=3, label='Pred')
    
    plt.xlabel('$x$')
    plt.ylabel('$C_f$')
    plt.title('Case '+str(i+1))
    plt.legend()
plt.tight_layout()

# %% result 
result_array = np.zeros((6,4))
for i in range(6):
    case_params = case_params_list[i]
    Re0 = torch.tensor(case_params['Re']).reshape(1,-1) / nn.Re_factor
    alpha0 = torch.tensor(case_params['alpha']/180*torch.pi).reshape(1,-1)
    b_up0 = torch.tensor(case_params['cst_up']).reshape(1,-1)
    b_down0 = torch.tensor(case_params['cst_down']).reshape(1,-1)
    state_para0 = torch.cat([Re0, alpha0, b_up0, b_down0], dim=1)
    
    nn.Predict_state_para(state_para0)
    
    result_array[i] = np.array([nn.Cl.item(),nn.Cd_p.item(),nn.Cd_f.item(),nn.Cd.item()])
    
result_array = np.round(result_array, 4)
print(result_array)

# %%
flowdata_path_list = ['result_Cp/NACA0012_400_150_Re100_AoA15_flow.dat',
                      'result_Cp/NACA4424_400_150_Re200_AoA11_flow.dat',
                      'result_Cp/RAE2822_400_150_Re500_AoA7_flow.dat',
                      'result_Cp/RAE5214_400_150_Re1000_AoA3_flow.dat',
                      'result_Cp/S2050_400_150_Re2500_AoA-1_flow.dat',
                      'result_Cp/S9000_400_150_Re5000_AoA-5_flow.dat']

fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
for i in range(3):  # 6 个算例
    for j, var in enumerate(['u', 'v', 'p']):  # 3 个变量
        case_params = case_params_list[i]
        Re0 = torch.tensor(case_params['Re']).reshape(1,-1) / nn.Re_factor
        alpha0 = torch.tensor(case_params['alpha']/180*torch.pi).reshape(1,-1)
        b_up0 = torch.tensor(case_params['cst_up']).reshape(1,-1)
        b_down0 = torch.tensor(case_params['cst_down']).reshape(1,-1)
        state_para0 = torch.cat([Re0, alpha0, b_up0, b_down0], dim=1)
        
        nn.Predict_state_para(state_para0)
        xx = nn.xx0.cpu().detach()
        yy = nn.yy.cpu().detach()
        uu = nn.U.cpu().detach()
        X = np.c_[xx.reshape(-1,1), yy.reshape(-1,1)]
        Data = np.loadtxt(flowdata_path_list[i], skiprows=1)[:,1:]
        kd_tree = KDTree(Data[:,:2])
        
        distances, indices = kd_tree.query(X, k=1)
        U_ref = Data[indices,:]
        uu_ref = U_ref[:,2:].reshape(nn.M, nn.N,3)[:,:,[1,2,0]]
        
        linestyle='solid'
        ax = axes[i, j]
        im = ax.contourf(xx, yy, uu[:,:,j], levels=20, cmap='jet')
        contour_lines = ax.contour(xx,yy,uu_ref[:,:,j], 20, colors='black', linewidths=1.0)
        for c in contour_lines.collections: c.set_linestyle(linestyle)
        r = 1
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title(f'${var}$')

ax = axes[0,0]
ax.text(-1.7,1.1,r'$\bf{(a)}$')
ax = axes[1,0]
ax.text(-1.7,1.1,r'$\bf{(b)}$')
ax = axes[2,0]
ax.text(-1.7,1.1,r'$\bf{(c)}$')

fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
for i in range(3):  # 6 个算例
    for j, var in enumerate(['u', 'v', 'p']):  # 3 个变量
        case_params = case_params_list[i+3]
        Re0 = torch.tensor(case_params['Re']).reshape(1,-1) / nn.Re_factor
        alpha0 = torch.tensor(case_params['alpha']/180*torch.pi).reshape(1,-1)
        b_up0 = torch.tensor(case_params['cst_up']).reshape(1,-1)
        b_down0 = torch.tensor(case_params['cst_down']).reshape(1,-1)
        state_para0 = torch.cat([Re0, alpha0, b_up0, b_down0], dim=1)
        
        nn.Predict_state_para(state_para0)
        xx = nn.xx0.cpu().detach()
        yy = nn.yy.cpu().detach()
        uu = nn.U.cpu().detach()
        X = np.c_[xx.reshape(-1,1), yy.reshape(-1,1)]
        Data = np.loadtxt(flowdata_path_list[i+3], skiprows=1)[:,1:]
        kd_tree = KDTree(Data[:,:2])
        
        distances, indices = kd_tree.query(X, k=1)
        U_ref = Data[indices,:]
        uu_ref = U_ref[:,2:].reshape(nn.M, nn.N,3)[:,:,[1,2,0]]
        
        linestyle='solid'
        ax = axes[i, j]
        im = ax.contourf(xx, yy, uu[:,:,j], levels=20, cmap='jet')
        contour_lines = ax.contour(xx,yy,uu_ref[:,:,j], 20, colors='black', linewidths=1.0)
        for c in contour_lines.collections: c.set_linestyle(linestyle)
        r = 1
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title(f'${var}$')

ax = axes[0,0]
ax.text(-1.7,1.1,r'$\bf{(d)}$')
ax = axes[1,0]
ax.text(-1.7,1.1,r'$\bf{(e)}$')
ax = axes[2,0]
ax.text(-1.7,1.1,r'$\bf{(f)}$')

# %% write flow field to tecplot
# with open('Flow.plt', 'w') as f:
#     f.write('variables=x,y,u,v,p\n')
#     f.write('zone I= %d, J=%d\n'%(nn.N, nn.M))
#     Data = torch.cat([nn.xx0.cpu().detach().reshape(-1,1), nn.yy.cpu().detach().reshape(-1,1), nn.U.cpu().reshape(-1,3)], dim=1)
#     for i in range(Data.shape[0]):
#         f.write('%f, %f, %f, %f, %f \n'%(Data[i,0],Data[i,1],Data[i,2],Data[i,3],Data[i,4]))
#     f.close()
