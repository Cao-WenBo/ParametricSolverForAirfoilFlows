import numpy as np
import matplotlib.pyplot as plt
import torch
from util import jacobian_trans, fwd_gradients, diff2d

class Net(torch.nn.Module):
    def __init__(self, layers, X, device):
        super(Net, self).__init__()
        self.X_mean = X.mean(0, keepdim=True)
        self.X_std = X.std(0, keepdim=True)

        self.num_layers = len(layers)
        self.layers = torch.nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i+1]))
        self.layers.to(device)   
        
    def forward(self, x, state_para):
        x = ((x - self.X_mean) / self.X_std) # z-score norm
        x = torch.cat([x, state_para], dim=1)
        for i in range(0, self.num_layers-1):
            x = self.layers[i](x)
            if i < self.num_layers-2:
                x = torch.tanh(x)
        return x
    
class Solver(): 
    def __init__(self, device, model, xx, yy, use_cst=True):
        self.device = device
        self.wall = {}

        # CST hyper-parameters
        self.degree = 5; self.n_1 = 0.5; self.n_2 = 1; 
        
        self.use_cst = use_cst
        self.Re_factor = 1000
        
        self.Preprocess(xx, yy)
        self.model = model
        
    def Preprocess(self, xx, yy):
        n_start_wall = 99 #66 #99
        
        self.xx0 = torch.tensor(xx.copy()).to(self.device).float()
        self.yy0 = torch.tensor(yy.copy()).to(self.device).float()
        self.X0 = torch.cat([self.xx0.reshape(-1,1),self.yy0.reshape(-1,1)],dim=1)
        
        #Initial airfoil
        self.ind_wall = torch.zeros(self.xx0[:,0].shape[0], dtype=torch.bool)
        self.ind_wall[n_start_wall:-n_start_wall] = True
        
        self.wall['x0'] = self.xx0[n_start_wall:-n_start_wall,0:1]
        self.wall['y0'] = self.yy0[n_start_wall:-n_start_wall,0:1]
        self.wall['X0'] = torch.cat([self.wall['x0'],self.wall['y0']], dim=1)
        self.mx = self.Cal_mx(self.wall['x0']+0.5).to(self.device)
        
        #computation domain
        self.M = xx.shape[0] #N_xi
        self.N = xx.shape[1] #N_eta

        self.xxi, self.eeta = torch.meshgrid(torch.arange(self.M), torch.arange(self.N))
        self.eeta, self.xxi= self.eeta.to(self.device).float(), self.xxi.to(self.device).float()
        self.XE = torch.cat([self.xxi.reshape(-1,1),self.eeta.reshape(-1,1)], dim=1)
        
        self.wall['XE'] = torch.cat([self.xxi[n_start_wall:-n_start_wall,0:1],self.eeta[n_start_wall:-n_start_wall,0:1]], dim=1)
        self.wall['XE_f'] = (self.wall['XE'][1:] + self.wall['XE'][:-1]) / 2
        
        #distance matrix for mesh deformation
        cdist = torch.sqrt((self.xx0-self.xx0[:,0:1])**2 + \
                                (self.yy0-self.yy0[:,0:1])**2)
        self.cdist = (cdist / cdist[:,-1:]).reshape(-1,1)
        cdist_ind = np.repeat(np.arange(cdist.shape[0]).reshape(-1,1), cdist.shape[1],1)
        self.cdist_ind = cdist_ind.reshape(-1,1)
        self.wall_ind = self.xxi.to(torch.int64)

    def Cal_mx(self, x_cor): 
        x_cor = x_cor.reshape(-1).cpu()
        num = x_cor.shape[0]
        mx = np.empty([num, self.degree+1])
        for index in range(self.degree+1):
            f = np.math.factorial(self.degree) / (np.math.factorial(index) * np.math.factorial(self.degree-index))
            mx[:, index] = f * np.power(x_cor, index+self.n_1) * np.power(1-x_cor, self.degree-index+self.n_2)
        mx = torch.tensor(mx).float()
        return mx
    
    def Predict_state_para(self, state_para0):
        

        state_para0 = state_para0.reshape(1,-1).to(self.device)
        

        Re0 = state_para0[0,0]
        alpha0 = state_para0[0,1]
        
        b_up0 = state_para0[:,2:8]
        b_down0 = state_para0[:,8:]
        N_up = int(self.wall['y0'].shape[0]/2)
        y = self.mx@b_down0.T; y[N_up:] = self.mx[N_up:]@b_up0.T 
        self.wall['y'] = y.detach()
        
        # Model 2, use y as input
        if self.use_cst == False:
            state_para0 = torch.cat([state_para0[0:1,:2], y.detach().T], dim=1)
        
        self.wall['X'] = torch.cat([self.wall['x0'], self.wall['y']], dim=1)
        self.wall['X0'] = torch.cat([self.wall['x0'], self.wall['y0']], dim=1)
        
        self.wall['X_f'] = (self.wall['X'][1:] + self.wall['X'][:-1]) / 2
        self.wall['dl_f'] = ((self.wall['X'][1:] - self.wall['X'][:-1]) ** 2).sum(1, keepdims=True) ** 0.5  # "_f" means face centre
        self.wall['tan_f'] = - (self.wall['X'][1:] - self.wall['X'][:-1]) / self.wall['dl_f']
        self.wall['nor_f'] = torch.cat([-self.wall['tan_f'][:,[1]], self.wall['tan_f'][:,[0]]], dim=1)
        
        dy = torch.zeros(self.xx0.shape[0], 1, device=self.device)
        dy[self.ind_wall] = (y - self.wall['y0'])
        yy = self.yy0 + (1 - self.cdist.reshape(self.M,self.N)) * dy  # Mesh deformation
        self.yy = yy.detach()
        
        nodes = torch.arange(-1,2,1, device=self.device, dtype=torch.long)
        Jac, Jac_inv = jacobian_trans(self.xx0.double(), self.yy.double(), nodes, boundary=(1,1))
        
        self.Jac_inv = Jac_inv.reshape(-1,2,2).float()
        self.Jac_inv_wall = Jac_inv[self.ind_wall,0].float()
        
        #Output flow field
        state_para = state_para0.repeat(self.XE.shape[0], 1)
        XE = self.XE*1
        self.U = self.model(XE, state_para).reshape(self.M,self.N,3)
        
        #Output Cp and Cf
        state_para = state_para0.repeat(self.wall['XE_f'].shape[0], 1)
        pred = self.model(self.wall['XE_f'], state_para)
        u = pred[:,0:1]; v = pred[:,1:2]; p = pred[:,2:3]
        self.Cp_f = 2*p
        
        state_para = state_para0.repeat(self.wall['XE'].shape[0], 1)
        XE = self.wall['XE']*1; XE.requires_grad = True
        U = self.model(XE, state_para)
        self.Cp = 2*U[:,2:3]
        
        dummy = torch.ones_like(U, requires_grad=True)
        gradsum = torch.autograd.grad(U, XE, dummy, create_graph=True, allow_unused=True)[0]
        U_xi = fwd_gradients(gradsum[:,0:1], dummy)
        U_eta = fwd_gradients(gradsum[:,1:2], dummy)
        U_xieta = torch.stack((U_xi, U_eta), dim=1)
        U_xy = self.Jac_inv_wall@U_xieta
        U_xy = (U_xy[1:] + U_xy[:-1]) / 2
        
        u_xy = U_xy[:,0:2,0]
        v_xy = U_xy[:,0:2,1]
        
        u_r = (u_xy * self.wall['nor_f']).sum(1, keepdim=True)
        v_r = (v_xy * self.wall['nor_f']).sum(1, keepdim=True)
        uth_r = u_r * self.wall['tan_f'][:,0:1] + v_r *self.wall['tan_f'][:,1:2]
        
        cf = - 1 / Re0 * uth_r / 0.5 / self.Re_factor
        self.Cf_f = cf # "_f" means face center
        
        Force = self.Cp_f*self.wall['dl_f']*self.wall['nor_f']
        self.Cl_p = (torch.cos(alpha0)*Force[:,1:2] - torch.sin(alpha0)*Force[:,0:1]).sum()
        self.Cd_p = (torch.sin(alpha0)*Force[:,1:2] + torch.cos(alpha0)*Force[:,0:1]).sum()
        
        Force = self.Cf_f*self.wall['dl_f']*self.wall['tan_f']
        self.Cl_f = (torch.cos(alpha0)*Force[:,1:2] - torch.sin(alpha0)*Force[:,0:1]).sum()
        self.Cd_f = (torch.sin(alpha0)*Force[:,1:2] + torch.cos(alpha0)*Force[:,0:1]).sum()
        
        self.Cl = self.Cl_p + self.Cl_f
        self.Cd = self.Cd_p + self.Cd_f


