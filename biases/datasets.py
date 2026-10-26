import torch
import numpy as np
import os
import networkx as nx
from torch.utils.data import Dataset
# from oil.utils.utils import Named, export
from biases.systems.rigid_body import RigidBody
from biases.systems.double_spherical_pendulum import DoubleSphericalPendulum
from biases.utils import rel_err, FixedSeedAll


class RigidBodyDataset(Dataset):
    space_dim = 3
    num_targets = 1

    def __init__(
        self,
        root_dir=None,
        body=DoubleSphericalPendulum(),
        n_systems=100,
        regen=False,
        chunk_len=5,
        angular_coords=False,
        seed=0,
        mode="train",
        n_subsample=None,
        stratified=False,
    ):
        super().__init__()
        with FixedSeedAll(seed):
            self.mode = mode
            root_dir = root_dir or os.path.expanduser(
                f"~/datasets/ODEDynamics/{self.__class__.__name__}/"
            )
            self.body = body
            filename = os.path.join(
                root_dir, f"trajectories_{body}_N{n_systems}_{mode}.pz"
            )
            if os.path.exists(filename) and not regen:
                ts, zs = torch.load(filename)
            else:
                ts, zs = self.generate_trajectory_data(n_systems)
                os.makedirs(root_dir, exist_ok=True)
                torch.save((ts, zs), filename)
            # Ts, Zs = self.chunk_training_data(ts, zs, chunk_len)

            # if n_subsample is not None:
            #     Ts, Zs = Ts[:n_subsample], Zs[:n_subsample]
            # Adapted -- geometry-aware training chunk selection
            if mode == "train" and stratified:
                if n_subsample is None:
                    raise ValueError(
                        "Stratified training requires n_subsample."
                    )

                Ts, Zs = self.chunk_training_data_stratified(
                    ts, zs,
                    chunk_len=chunk_len,
                    n_subsample=n_subsample,
                )

            else:
                Ts, Zs = self.chunk_training_data(
                    ts, zs,
                    chunk_len
                )

                if n_subsample is not None:
                    Ts, Zs = Ts[:n_subsample], Zs[:n_subsample]
                    
            self.Ts, self.Zs = Ts.float(), Zs.float()
            self.seed = seed
            if angular_coords: 
                N, T = self.Zs.shape[:2]
                flat_Zs = self.Zs.reshape(N * T, *self.Zs.shape[2:])
                self.Zs = self.body.global2bodyCoords(flat_Zs.double())
                print(rel_err(self.body.body2globalCoords(self.Zs), flat_Zs))
                self.Zs = self.Zs.reshape(N, T, *self.Zs.shape[1:]).float()

    def __len__(self):
        return self.Zs.shape[0]

    def __getitem__(self, i):
        return (self.Zs[i, 0], self.Ts[i]), self.Zs[i]

    def generate_trajectory_data(self, n_systems, bs=10000):
        """ Returns ts: (n_systems, traj_len) zs: (n_systems, traj_len, z_dim) """

        def base_10_to_base(n, b):
            """Writes n (originally in base 10) in base `b` but reversed"""
            if n == 0:
                return '0'
            nums = []
            while n:
                n, r = divmod(n, b)
                nums.append(r)
            return list(nums)

        batch_sizes = base_10_to_base(n_systems, bs)
        n_gen = 0
        t_batches, z_batches = [], []
        for i, batch_size in enumerate(batch_sizes):
            if batch_size == 0:
                continue
            batch_size = batch_size * (bs**i)
            print(f"Generating {batch_size} more chunks")
            z0s = self.sample_system(batch_size)
            ts = torch.arange(
                0, self.body.integration_time, self.body.dt, device=z0s.device, dtype=z0s.dtype
            )
            new_zs = self.body.integrate(z0s, ts)
            t_batches.append(ts[None].repeat(batch_size, 1))
            z_batches.append(new_zs)
            n_gen += batch_size
            print(f"{n_gen} total trajectories generated for {self.mode}")
        ts = torch.cat(t_batches, dim=0)[:n_systems]
        zs = torch.cat(z_batches, dim=0)[:n_systems]
        return ts, zs

    def chunk_training_data(self, ts, zs, chunk_len):
        """ Randomly samples chunks of trajectory data, returns tensors shaped for training.
        Inputs: [ts (batch_size, traj_len)] [zs (batch_size, traj_len, *z_dim)]
        outputs: [chose_ts (batch_size, chunk_len)] [chosen_zs (batch_size, chunk_len, *z_dim)]"""
        n_trajs, traj_len, *z_dim = zs.shape
        n_chunks = traj_len // chunk_len
        # Cut each trajectory into non-overlapping chunks
        chunked_ts = torch.stack(ts.chunk(n_chunks, dim=1))
        chunked_zs = torch.stack(zs.chunk(n_chunks, dim=1))
        # From each trajectory, we choose a single chunk randomly
        chunk_idx = torch.randint(0, n_chunks, (n_trajs,), device=zs.device).long()
        chosen_ts = chunked_ts[chunk_idx, range(n_trajs)]
        chosen_zs = chunked_zs[chunk_idx, range(n_trajs)]
        return chosen_ts, chosen_zs

    def chunk_training_data_stratified(self, Ts, Zs, chunk_len, n_subsample):
        """
        Adapted -- geometry-aware training chunk selection.
        
        Selects chunks according to the maximum theta_1 reached within
        each chunk, while allowing at most one selected chunk per source 
        trajectory.
        
        Validation/test data should continue using Finzi's original
        chunk_training_data().
        """

        n_trajs, traj_len, *z_dim = Zs.shape
        n_chunks = traj_len // chunk_len

        chunked_ts = torch.stack(Ts.chunk(n_chunks, dim=1))
        chunked_zs = torch.stack(Zs.chunk(n_chunks, dim=1))

        # Position of bob 1:
        # (..., state=[x,v], body, xyz)
        r1 = chunked_zs[:, :, :, 0, 0, :]

        cos_theta1 = (
            r1[..., 2] / 
            torch.linalg.vector_norm(r1, dim=-1)
        ).clamp(-1.0, 1.0)

        theta1 = torch.rad2deg(torch.acos(cos_theta1))

        # Characterize each canditate chunk by the largest theta_1
        # reached anywhere inside its 5 states.
        theta1_max = theta1.max(dim=2).values

        bins = [
            (0.0, 90.0, 100),
            (90.0, 100.0, 20),
            (100.0, 110.0, 20),
            (110.0, 120.0, 20),
            (120.0, 130.0, 15),
            (130.0, 140.0, 15),
            (140.0, 180.1, 10),
        ]

        assert sum(q for _, _, q in bins) == n_subsample

        selected = []
        used_trajs = set()

        generator = torch.Generator()
        generator.manual_seed(0)

        for lo, hi, quota in bins:

            candidates = torch.nonzero(
                (theta1_max >= lo) & (theta1_max < hi),
                as_tuple=False
            )

            # Randomize candidate order reproducibly
            order = torch.randperm(
                len(candidates),
                generator=generator
            )

            count = 0

            for idx in order:
                chunk_idx, traj_idx = candidates[idx].tolist()

                # Keep one selected chunk per analytical trajectory
                if traj_idx in used_trajs:
                    continue

                selected.append((chunk_idx, traj_idx))
                used_trajs.add(traj_idx)

                count += 1

                if count == quota:
                    break

            if count < quota:
                raise RuntimeError(
                    f"Not enough unique trajectories for "
                    f"theta1 bin [{lo}, {hi}): "
                    f"requested {quota}, found {count}"
                )

        selected_ts = torch.stack([
            chunked_ts[chunk_idx, traj_idx]
            for chunk_idx, traj_idx in selected
        ])

        selected_zs = torch.stack([
            chunked_zs[chunk_idx, traj_idx]
            for chunk_idx, traj_idx in selected
        ])

        return selected_ts, selected_zs

    def sample_system(self, N):
        """"""
        return self.body.sample_initial_conditions(N)

    