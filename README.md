# MultiRobot-Distribution-Matching

Official implementation of the paper **"Optimal Transport-Based Decentralized Multi-Agent Distribution Matching"**, published in **IEEE Transactions on Automatic Control (TAC), 2026**.

This repository provides robust control algorithms for multi-robot systems to achieve target distribution matching using **Optimal Transport** theory.

## Links
* **Paper (IEEE)**: https://doi.org/10.1109/TAC.2026.3668445
* **Paper (arXiv)**: https://arxiv.org/abs/2601.00548
---

## Repository Overview
We provide implementations for both centralized and decentralized scenarios:

1.  **centralized_main.py**: Solves the global optimal transport problem sequentially to determine the optimal target position for each robot.
2.  **decentralized_main.py**: Implements a memory-augmented min-consensus algorithm for scenarios with limited local communication.

### Core Dynamics
This implementation applies optimal control based on **linear time-invariant (LTI) system dynamics** to achieve efficient convergence to the target distribution within a finite horizon **$T$**:
$$U^* = \Phi_T^T \Gamma_T^{-1} (y^* - A^T x_k)$$

### Nonlinear Control-Affine Dynamics
For the development of **control-affine nonlinear dynamics**, please refer to the main paper.
---

## Installation & Usage
### Requirements
* Python 3.8+
* numpy
* matplotlib

### Run Simulations
```bash
# Run centralized simulation
python centralized_main.py

# Run decentralized simulation
python decentralized_main.py
