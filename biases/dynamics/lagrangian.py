import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch.autograd import grad
from torch import Tensor
from typing import Callable


class ConstrainedLagrangianDynamics(nn.Module):
    """Defines the Constrained Hamiltonian dynamics given a Hamiltonian and 
    gradients of constraints.
    
    Args: 
        V: A callable function that takes in x and returns the potential
        DPhi:
        wgrad: If True, the dynamics can be backproped.
    """

    def __init__(
        self,
        V,Minv, # And ultimately A
        DPhi: Callable[[Tensor], Tensor],
        shape,
        wgrad: bool = True,
    ):
        super().__init__()
        self.V = V
        self.Minv = Minv
        self.DPhi = DPhi 
        self.shape = shape
        self.wgrad = wgrad
        self.nfe = 0

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        """Computes a batch of `NxD` time derivatives of the state `z` at time `t`
        Args:
            t: Scalar Tensor of the current time
            z: bs x 2nd Tensor of the bs different states in D=2nd dimensions
        """
        assert (t.ndim == 0) and (z.ndim == 2)
        bs,n,d = z.shape[0],*self.shape
        self.nfe += 1
        with torch.enable_grad():
            x, v = z.reshape(bs,2,n,d).unbind(dim=1)                    # (bs, n, d)
            x = torch.zeros_like(x, requires_grad=True) + x
            dV = torch.autograd.grad(self.V(x).sum(),x,create_graph=self.wgrad)[0]
            f = -self.Minv(dV).reshape(bs,n*d)                           # (bs,nd)
            DPhi = self.DPhi(x,v)                                   # (bs,2,n,d,2,C)
            if DPhi.shape[-1]!=0:
                G = DPhi[:,0,:,:,0,:].reshape(bs,n*d,-1)                    # (bs,nd, C)
                GT = G.permute(0,2,1)                                       # (bs,C, nd)
                GdotT = DPhi[:,0,:,:,1,:].reshape(bs,n*d,-1).permute(0,2,1) # (bs,nd, C)
                MinvG = self.Minv(G.reshape(bs,n,-1)).reshape(G.shape)      # (bs,nd, C)
                GTMinvG = GT@MinvG                                          # (bs, C, C)
                GTf = (GT@f.unsqueeze(-1)).squeeze(-1)                      # (bs, C)
                violation = (GdotT@v.reshape(bs,n*d,1)).squeeze(-1)         # (bs, C)
                total_violation = (GTf+violation).unsqueeze(-1)             # (bs,nd, 1)
                lambdas = (MinvG@torch.linalg.solve(GTMinvG, total_violation)).squeeze(-1) # (bs,nd)
                vdot = f - lambdas                                          # (bs,nd)
            else: vdot = f
            dynamics = torch.cat([v.reshape(bs,n*d),vdot],dim=-1)       # (bs,2nd)
        return dynamics