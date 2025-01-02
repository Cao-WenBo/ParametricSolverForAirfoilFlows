# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import io
import os

def fwd_gradients(Y, x):
    dummy = torch.ones_like(Y)
    G = torch.autograd.grad(Y, x, dummy, create_graph= True)[0]
    return G

def load_class_name(case_prefix_list, base_path='./'):
    max_epoch_files = []


    for case_prefix in case_prefix_list:
        case_folders = [f for f in os.listdir(base_path) if f.startswith(case_prefix)]

        for case_folder in case_folders:
            case_folder_path = os.path.join(base_path, case_folder)

            if os.path.isdir(case_folder_path):
                class_files = [f for f in os.listdir(case_folder_path) if f.endswith('.pkl')]
                if not class_files: print(case_prefix+'no file')

                epoch_numbers = []
                for file in class_files:
                    epoch = int(file.split('.')[0]) 
                    epoch_numbers.append((epoch, file))

                if epoch_numbers:
                    max_epoch_file = max(epoch_numbers, key=lambda x: x[0])[1]
                    max_epoch_files.append(os.path.join(case_folder_path, max_epoch_file))
    nn_list = []
    for file in max_epoch_files:
        nn_list.append(load_class(file))

    return nn_list

def load_class(file_name, device='cuda:0'):
    with open(file_name, 'rb') as file: 
        class Device_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location=device)
                else:
                    return super().find_class(module, name)
        obj = Device_Unpickler(file).load()
    obj.device = device
    return obj

def save_class(obj, file_name, device='cuda:0'):
    with open(file_name,'wb') as file: 
        file.write(pickle.dumps(obj)); file.close()

def coeff(nodes, order=1):
    '''
    This function is used to calculate the differential coefficient of given nodes
    '''
    m = len(nodes)
    device=nodes.device

    factor, pownodes = torch.ones(m, device=device), torch.ones(m, m, device=device)
    b = torch.zeros(m, 1, device=device)
    b[order] = 1

    for i in range(1, m):
        factor[i] = i * factor[i-1]
    for i in range(1,m):
        pownodes[:,i] = nodes * pownodes[:,i-1]
        
    A = pownodes / factor
    A = A.T
    x = torch.linalg.solve(A, b)
    return x

def boundary_2nd(u, nodes, du=0, dim=0, loc=['left', 'right']):
    if dim == 1: 
        u = u.T
    
    if 'left' in loc:
        ind_l = nodes[nodes>=0]
        coeff_l = coeff(ind_l)
        u[0:1] = - (u[ind_l[1:]]*coeff_l[ind_l[1:]]).sum(0, keepdim=True) / coeff_l[0]
    if 'right' in loc:
        ind_r = nodes[nodes<=0]
        coeff_r = coeff(ind_r)
        u[-1:] = - (u[ind_r[:-1]-1] * coeff_r[ind_r[:-1]-1]).sum(0, keepdim=True) / coeff_r[-1]
    
    if dim == 1: 
        u = u.T
    return u

def jacobian_trans(x, y, nodes, boundary=(0,0)):
    Xxi = diff2d(x, nodes, dim=0, boundary=boundary[0])
    Xeta = diff2d(x, nodes, dim=1, boundary=boundary[1])
    Yxi = diff2d(y, nodes, dim=0, boundary=boundary[0])
    Yeta = diff2d(y, nodes, dim=1, boundary=boundary[1])
    
    jac = torch.stack([Xxi, Yxi, Xeta, Yeta], dim=2).view(x.shape[0],x.shape[1],2,2)
    jac_inv = torch.inverse(jac)
    return jac, jac_inv

def diff2d_jac_inv(jac_inv, u, nodes, boundary=(0,0)):
    u_xi = diff2d(u, nodes, dim=0, boundary=boundary[0])
    u_eta = diff2d(u, nodes, dim=1, boundary=boundary[1])
    u_xieta = torch.stack([u_xi, u_eta], dim=2).view(u.shape[0],u.shape[1],2,1)
    u_xy = (jac_inv@u_xieta).squeeze()
    return u_xy

def diff2d_xy(x, y, u, nodes, boundary=(0,0)):
    jac, jac_inv = jacobian_trans(x, y, nodes, boundary=boundary)
    u_xy = diff2d_jac_inv(jac_inv, u, nodes, boundary=boundary)
    return u_xy

def diff2d(u, nodes, dim=0, order=1, boundary=0):
    # boundary=0: The boundaries use the same number of nodes
    # boundary=1: The boundaries use smaller number of nodes
    # boundary=2: Periodic boundary
    if dim == 1:
        u = u.T
    (M, N) = u.shape
    coeff_u = coeff(nodes, order)
    m, n = nodes[0], nodes[-1]
    du = torch.zeros_like(u)
    
    if boundary == 0 or boundary == 1:
        for i in range(len(nodes)):
            du[-m:M-n] = du[-m:M-n] + coeff_u[i] * u[-m+nodes[i]:M-n+nodes[i]]
            
        for i in range(len(nodes)):
            if nodes[i] == 0: 
                continue
            elif nodes[i] < 0:
                ind = nodes[i]-m
                if boundary == 0: nodes_bound = nodes - nodes[i]
                if boundary == 1: nodes_bound = nodes[nodes >= -ind]
                coeff_bound = coeff(nodes_bound, order)
                nodes_ind = ind + nodes_bound
                du[ind] = du[ind] + (coeff_bound * u[nodes_ind]).sum(0)
            else:
                ind = nodes[i]-n + M-1
                if boundary == 0: nodes_bound = nodes - nodes[i]
                if boundary == 1: nodes_bound = nodes[nodes <= M-1-ind]
                coeff_bound = coeff(nodes_bound, order)
                nodes_ind = ind + nodes_bound
                du[ind] = du[ind] + (coeff_bound * u[nodes_ind]).sum(0)

    elif boundary == 2:
        u = torch.cat([u[m:], u, u[:n]]); (M, N) = u.shape
        du = torch.zeros_like(u)
        for i in range(len(nodes)):
            du[-m:M-n] = du[-m:M-n] + coeff_u[i] * u[-m+nodes[i]:M-n+nodes[i]]
        du = du[-m:M-n]
        
    if dim == 1:
        du = du.T
    return du
