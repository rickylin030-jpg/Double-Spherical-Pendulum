import torch
import networkx as nx
import numpy as np
# from oil.utils.utils import export,FixedNumpySeed
from biases.systems.rigid_body import RigidBody, BodyGraph, project_onto_constraints
# from animation import Animation
import copy


class DoubleSphericalPendulum(RigidBody):
    d=3
    dt=0.03
    integration_time=3
    def __init__(self, m1=1., m2=1., R1=1., R2=1., g=9.8):
        self.body_graph = BodyGraph()#nx.Graph()
        self.m1 = m1
        self.m2 = m2
        self.R1 = R1 
        self.R2 = R2
        self.g = g

        self.body_graph.add_extended_nd(0, m=m1, d=0, tether=(torch.zeros(3), R1))
        self.body_graph.add_extended_nd(1, m=m2, d=0)
        self.body_graph.add_edge(0, 1, l=R2)
        self.n = 2
        self.D = 4
        self.angular_dims = range(4)

    def body2globalCoords(self, angles_omega):
        d = 3
        n = len(self.body_graph.nodes)
        N = angles_omega.shape[0]
        pvs = torch.zeros(N, 2, n, d,device=angles_omega.device,dtype=angles_omega.dtype)
        global_position_velocity = torch.zeros(N, 2, d,device=angles_omega.device,dtype=angles_omega.dtype)
        length = self.body_graph.nodes[0]["tether"][1]
        global_position_velocity[:, 0, :] = self.body_graph.nodes[0]["tether"][0][None]
        global_position_velocity += self.joint2cartesian(length, angles_omega[..., 0])
        pvs[:, :, 0] = global_position_velocity
        for (_, j), length in nx.get_edge_attributes(self.body_graph, "l").items():
            global_position_velocity += self.joint2cartesian(length, angles_omega[..., j])
            pvs[:, :, j] = global_position_velocity
        return pvs

    def joint2cartesian(self, length, angle_omega):
        position_vel = torch.zeros(angle_omega.shape[0], 2, 3, device=angle_omega.device, dtype=angle_omega.dtype)
        position_vel[:, 0, 0] = length * angle_omega[:, 0].sin() * angle_omega[:, 1].cos()
        position_vel[:, 1, 0] = length * (angle_omega[:, 2] * angle_omega[:, 0].cos() * angle_omega[:, 1].cos() - angle_omega[:, 3] * angle_omega[:, 0].sin() * angle_omega[:, 1].sin())
        position_vel[:, 0, 1] = length * angle_omega[:, 0].sin() * angle_omega[:, 1].sin()
        position_vel[:, 1, 1] = length * (angle_omega[:, 2] * angle_omega[:, 0].cos() * angle_omega[:, 1].sin() + angle_omega[:, 3] * angle_omega[:, 0].sin() * angle_omega[:, 1].cos())
        position_vel[:, 0, 2] = length * angle_omega[:, 0].cos()
        position_vel[:, 1, 2] = -length * angle_omega[:, 2] * angle_omega[:, 0].sin()
        return position_vel

    def sample_initial_conditions(self,N):
        n = len(self.body_graph.nodes)
        angles_and_angvel = torch.randn(N, 4, n)  # (N,4,n)
        z = self.body2globalCoords(angles_and_angvel)
        #z = torch.randn(N,2,n,3)
        z[:,0] += .2*torch.randn(N,n,3)
        z[:,1] = (.5*z[:,1] + .4*torch.randn(N,n,3))*3
        try: return project_onto_constraints(self.body_graph,z,tol=1e-5)
        except OverflowError: return self.sample_initial_conditions(N)

    def potential(self, x):
        return -self.g * (self.M @ x)[..., 2].sum(1)
        # self.arg_string = f"n{links}{'b' if beams else ''}m{m or 'r'}l{l or 'r'}"
        # assert not beams, "beams temporarily not supported"
        # with FixedNumpySeed(0):
        #     ms = [.6+.8*np.random.rand() for _ in range(links)] if m is None else links*[m]
        #     ls = [.6+.8*np.random.rand() for _ in range(links)] if l is None else links*[l]
        # self.ms = copy.deepcopy(ms)
        # self.body_graph.add_extended_nd(0, m=ms.pop(), d=0,tether=(torch.zeros(2),ls.pop()))
        # for i in range(1, links):
        #     self.body_graph.add_extended_nd(i, m=ms.pop(), d=0)
        #     self.body_graph.add_edge(i - 1, i, l=ls.pop())
        # self.D =self.n = links
        # self.angular_dims = range(links)

    def __str__(self):
        # return f"{self.__class__}{self.arg_string}"
        return self.__class__.__name__ # TEMPORARY

    def __repr__(self):
        return str(self)

    # @property
    # def animator(self):
    #     return PendulumAnimation
