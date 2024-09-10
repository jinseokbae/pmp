Official Implementation of "PMP: Learning to Physically Interact with Environments using Part-wise Motion Priors" (SIGGRAPH 2023) ([paper](https://arxiv.org/abs/2305.03249), [video](https://www.youtube.com/watch?v=WdLGvKdNG-0&t=21s), [talk](https://www.youtube.com/watch?v=WzvFRI5FxRI))

# Status

## Released
- [O] Assets
    - [O] Deepmimic-MPL Humanoid
    - [O] Objects for interaction
    - [ ] Retargeted motion data (check for the license)

- [ ] Simulation Configuration Files
    - [O] `.yaml` files for whole-body and hand-only gyms
    - [ ] documentations about details

## Todo
- [ ] Shell script to install all external dependencies

- [ ] Retargeting pipeline (Mixamo to Deepmimic-MPL Humanoid)

- [ ] Whole-body Gym : training hand-equipped humanoid
    - [ ] Model (Train / Test)
        - pretrained weights
    - [ ] Environments

- [ ] Hand-only Gym : training one hand to grab a bar
    - [ ] Model (Train / Test)
        - pretrained weight
        - expert trajectories
    - [ ] Environment


**Note**) I'm currently focusing on the other projects mainly so this repo will be updated slowly.
In case you require early access to the full implementation, I can share unofficial versions via e-mail.
Please contact me via e-mail.

# Installation
This code is based on [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym).
Please run installation code and create a conda environment following the instruction in Isaac Gym Preview 4.
We assume the name of conda environment is `pmp_env`.
Then, run the following script.

```shell
conda activate pmp_env
cd pmp
pip install -e .
```

# Acknowledgement
## Codebase
This code is based on the official release of [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs).
Especially, this code largely borrows implementations of `AMP` in the original codebase ([paper](https://arxiv.org/abs/2104.02180), [code](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/humanoid_amp.py)).

## Humanoid
Our whole-body agent is modified from the humanoid in [Deepmimic](https://arxiv.org/abs/1804.02717). 
We replace the sphere-shaped hands of the original humanoid with the hand from [Modular Prosthetic Limb (MPL)](https://dl.acm.org/doi/abs/10.1109/HUMANOIDS.2015.7363441).

## Motion data
We use [Mixamo](https://www.mixamo.com) animation data for training part-wise motion prior.
We retarget mixamo animation data into our whole-body humanoid using the similar process used in the [original codebase](https://github.com/isaac-sim/IsaacGymEnvs).




