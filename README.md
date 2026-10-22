# Double Spherical Pendulum

A computational study of the **double spherical pendulum (DSP)** — two point masses connected by rigid massless rods, each bob free to move on a sphere — combining analytical mechanics, numerical integration, and physics-informed machine learning.

The system has four generalized coordinates $(\theta_1, \phi_1, \theta_2, \phi_2)$ and an eight-dimensional state space. It conserves the Hamiltonian (time-translation symmetry) and the $z$-component of angular momentum (rotational symmetry about the vertical axis), yet still exhibits chaotic motion at large perturbation amplitudes — which makes it a good testbed for both classical numerical methods and learned dynamics models.

## What's here

- **Analytical mechanics** — Lagrangian and Hamiltonian formulations, Euler–Lagrange and Hamilton's equations, generalized momenta, Legendre transformation, and a numerical check that both formulations produce the same trajectories.
- **Relative equilibria and normal modes** — aligned and anti-aligned double-conical solutions, linearization about them, the quadratic eigenvalue problem $(-\omega^2 M + i\omega B + K)\mathbf{a} = 0$, and modal vector fields.
- **Chaos diagnostics** — sensitivity to initial conditions as perturbation amplitude grows, state-space projections, and Poincaré sections.
- **Integrator benchmarking** — six SciPy methods (RK23, RK45, DOP853, LSODA, Radau, BDF) compared on work–precision and energy-conservation grounds against a high-accuracy DOP853 reference.
- **Conserved-quantity and Liouville checks** — numerical verification of $\mathcal{H}$ and $L_z$ conservation, plus a phase-space-volume demonstration.
- **Constrained Lagrangian Neural Networks (CLNN)** — three training experiments learning the DSP dynamics in Cartesian coordinates with explicit holonomic constraints.
- **Lagrangian Neural Network (LNN)** — a JAX implementation adapted from Cranmer et al. to the four-coordinate DSP state.

## Repository layout

```
.
├── pl_trainer.py                       # PyTorch Lightning training/eval entry point for CLNN
├── biases/                             # Constrained-dynamics library (see Attribution)
│   ├── systems/
│   │   ├── rigid_body.py               # RigidBody, BodyGraph, constraint projection
│   │   └── double_spherical_pendulum.py# DSP body: masses, rods, body graph, potential
│   ├── models/
│   │   ├── constrained_lnn.py          # CLNN / CL network definitions
│   │   └── utils.py                    # FCtanh, Linear, Reshape layers
│   ├── dynamics/
│   │   ├── lagrangian.py               # ConstrainedLagrangianDynamics
│   │   └── hamiltonian.py              # ConstrainedHamiltonianDynamics, EuclideanT
│   ├── datasets.py                     # RigidBodyDataset, chunking, stratified sampling
│   ├── animation.py                    # 3D trajectory animation helpers
│   └── utils.py                        # rel_err, FixedSeedAll
├── notebooks/
│   ├── dsp.ipynb                       # Main investigation (sections 1–5)
│   └── dsp_lnn.ipynb                   # JAX Lagrangian Neural Network for the DSP
└── experiments/
    └── DoubleSphericalPendulum/CLNN/   # Offline W&B runs: checkpoints, args, test logs
```

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**PyTorch.** `requirements.txt` installs the default CPU/CUDA wheel for your platform. The experiments here were run against `torch==2.13.0+cu130`; for a specific CUDA build, install torch first from the [official index](https://pytorch.org/get-started/locally/) and then the rest of the requirements.

**JAX.** Only `notebooks/dsp_lnn.ipynb` needs JAX. The CPU wheel in `requirements.txt` is enough to run it; for GPU, follow the [JAX install guide](https://docs.jax.dev/en/latest/installation.html).

**Weights & Biases.** Training logs through W&B. To run without an account:

```bash
export WANDB_MODE=offline      # Windows PowerShell: $env:WANDB_MODE = "offline"
```

The runs committed under `experiments/` were produced in offline mode.

## Usage

### Notebooks

`notebooks/dsp.ipynb` is the main document and reads top to bottom; it uses `%matplotlib widget`, so run it in JupyterLab with `ipympl` installed. `notebooks/dsp_lnn.ipynb` is self-contained and does not require the `lnn` package.

The CLNN evaluation section locates the committed W&B runs relative to the repository root, so it works on any checkout. If you launch Jupyter from somewhere other than the repository, point it at the right place:

```bash
export DSP_REPO_ROOT=/path/to/double_spherical_pendulum
```

### Training a CLNN

`pl_trainer.py` generates trajectory data from the analytical DSP dynamics, trains the model, and evaluates rollouts against the true system. Training data is cached under `~/datasets/ODEDynamics/RigidBodyDataset/`; pass `--regen` to rebuild it.

```bash
python pl_trainer.py \
  --body-class DoubleSphericalPendulum \
  --network-class CLNN \
  --n-train 200 --n-val 50 --n-test 50 \
  --n-epochs 500 --n-epochs-per-val 50 \
  --batch-size 50
```

Add `--n-gpus 0` to train on CPU. Checkpoints, `args.csv`, and `test_log.pt` are written under `experiments/DoubleSphericalPendulum/<network-class>/`.

### The three CLNN experiments

All three use the same architecture — 3 hidden layers, 256 units, 133,635 trainable parameters, AdamW with cosine annealing, `lr=3e-3`, `chunk_len=5` — and differ only in training data and optimization budget.

| | Sampling | Train chunks | Epochs | Updates | Batch | Extra flag |
|---|---|---|---|---|---|---|
| **Exp1** | random | 200 | 500 | 2,000 | 50 | — |
| **Exp2** | geometrically stratified | 200 | 500 | 2,000 | 50 | `--stratified` |
| **Exp3** | random | 800 | 2,000 | 8,000 | 200 | — |

Exp1 vs. Exp2 isolates the sampling strategy at fixed data quantity and training length. Exp1 vs. Exp3 varies data quantity and optimization duration together, so their individual effects cannot be separated.

**Stratified sampling** (`--stratified`, implemented in `biases/datasets.py`) selects training chunks so that configurations with large first-bob polar angles are better represented. Under random sampling roughly 95% of chunks stay below $\theta_1^{\max} = 90°$; stratification raises the share at or above $90°$ to about 50% without increasing the dataset size.

### Results summary

Evaluated over 50 analytical reference trajectories across a 2.97 s horizon, using relative state-space error and instantaneous MAE with 95% bootstrap confidence intervals:

- Exp1 accumulates error fastest and stays well above the other two for most of the horizon.
- Exp2 and Exp3 both improve substantially on Exp1; their confidence intervals overlap, so neither is clearly better than the other.
- Matching stratified sampling on the same 200 chunks (Exp2) to a 4× larger randomly-sampled dataset trained 4× longer (Exp3) suggests that *where* the training chunks sit in configuration space matters as much as how many there are.

On the classical side, DOP853, LSODA, and RK45 gave the best work–precision tradeoffs, with DOP853 reaching an RMS relative energy error of $1.37 \times 10^{-10}$ at the strictest tolerances.

## Attribution

- The `biases/` package is adapted from the reference implementation accompanying Finzi, Wang & Wilson, *Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints* (NeurIPS 2020). The `DoubleSphericalPendulum` system, the stratified-sampling path in `datasets.py`, and the PyTorch Lightning 2.x compatibility changes in `pl_trainer.py` are additions made here; edits to upstream code are marked with `# Adapted` comments.
- `notebooks/dsp_lnn.ipynb` adapts the Lagrangian Neural Network of Cranmer et al. (2020) from the planar double pendulum to the four-coordinate DSP, keeping the original architecture, Euler–Lagrange formulation, optimizer, and pseudoinverse-based acceleration solve.

- The `Pendulum`, `AnimationPendulumPlot` and plotting helpers in `notebooks/dsp.ipynb` are adapted from the OSU Physics 5300 course notebooks ([jdbrice/5300-JB](https://github.com/jdbrice/5300-JB)), credited to Dick Furnstahl. They were extended from one or two planar coordinates to the DSP's four generalized coordinates and subclassed as `HamiltonianPendulum`, `HamiltonianAnimationPendulumPlot` and `NormalModeHamiltonianPendulum`.

See [NOTICE](NOTICE) for the per-file provenance and the upstream terms.

## References

1. Taylor, J. R. (2005). *Classical Mechanics* (1st ed.). University Science Books. Problem 7.40; Ch. 13 §13.7, "Liouville's Theorem."
2. Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S. (2020). *Lagrangian Neural Networks*. ICLR 2020 Workshop DeepDiffEq. https://arxiv.org/abs/2003.04630
3. Finzi, M., Wang, K. A., & Wilson, A. G. (2020). *Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints*. NeurIPS 2020. https://arxiv.org/abs/2010.13581
4. Furnstahl, R. J. (2019). *Liouville Theorem Visualization*. Computational physics teaching notebook, revised March 16, 2019. Adapted for the reduced normal-mode Hamiltonian of the double spherical pendulum.

## License

The original work in this repository — the DSP system definition, the stratified-sampling additions, the Lightning 2.x compatibility changes, and the notebooks — is released under the [MIT License](LICENSE).

Parts of the repository derive from third-party projects and are **not** covered by that license. [NOTICE](NOTICE) lists them file by file. In summary:

| Component | Upstream | Terms |
|---|---|---|
| `biases/` (constrained-dynamics library) | [mfinzi/constrained-hamiltonian-neural-networks](https://github.com/mfinzi/constrained-hamiltonian-neural-networks) | No license published; included by permission of the authors, with attribution ([NOTICE](NOTICE)) |
| `notebooks/dsp_lnn.ipynb` (LNN implementation) | [MilesCranmer/lagrangian_nns](https://github.com/MilesCranmer/lagrangian_nns) | [Apache License 2.0](LICENSES/Apache-2.0-lagrangian_nns.txt) |
| `notebooks/dsp.ipynb` (pendulum, animation and plotting scaffolding) | [jdbrice/5300-JB](https://github.com/jdbrice/5300-JB), OSU Physics 5300, notebooks credited to Dick Furnstahl | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) per its LICENSE file |
