# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import logging
import os
from datetime import datetime, timedelta, timezone

import gym
import hydra

# noinspection PyUnresolvedReferences
import isaacgym
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

import signal
import sys
import secrets
import string

def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict["params"]["config"]

    train_cfg["device"] = cfg.rl_device

    train_cfg["full_experiment_name"] = cfg.train.params.config.get("full_experiment_name")

    print(f"Using rl_device: {cfg.rl_device}")
    print(f"Using sim_device: {cfg.sim_device}")
    print(train_cfg)

    try:
        model_size_multiplier = config_dict["params"]["network"]["mlp"]["model_size_multiplier"]
        if model_size_multiplier != 1:
            units = config_dict["params"]["network"]["mlp"]["units"]
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(
                f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}'
            )
    except KeyError:
        pass

    return config_dict

def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits."""
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))

@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    from rl_games.algos_torch import model_builder
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    import isaacgymenvs
    from isaacgymenvs.learning import (
        hand_continuous,
        hand_players,
        hand_models,
        hand_network_builder
    )
    from isaacgymenvs.utils.rlgames_utils import ComplexObsRLGPUEnv, MultiObserver, RLGPUAlgoObserver, RLGPUEnv
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

        # search for hash_code
        exp_dir = cfg.checkpoint.split('/')[:-2]
        if not os.path.isfile(os.path.join('/'.join(exp_dir), 'hash_code.txt')) and not cfg.task.env.test:
            raise ValueError("Hashed code is needed to resume training!")
        cfg.train.params.config.name = exp_dir[-1]
        
    # when training from scratch
    elif not cfg.task.env.test:
        hash_code = generate_id()
        cfg.train.params.config.name += '_' + hash_code
        os.makedirs(f'runs/{cfg.train.params.config.name}')
        with open(f'runs/{cfg.train.params.config.name}/hash_code.txt', 'w') as f:
            f.write(hash_code)

    cfg_dict = omegaconf_to_dict(cfg)
    # always naming dummy for simple test
    if cfg.task.env.test:
        cfg.train.params.config.name = "dummy"
    cfg.train.params.config.full_experiment_name = cfg.train.params.config.name
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        return envs

    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: create_isaacgym_env(**kwargs),
        },
    )

    ige_env_cls = isaacgym_task_map[cfg.task_name]
    dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, "dict_obs_cls") and ige_env_cls.dict_obs_cls else False

    if dict_cls:

        obs_spec = {}
        actor_net_cfg = cfg.train.params.network
        obs_spec["obs"] = {
            "names": list(actor_net_cfg.inputs.keys()),
            "concat": not actor_net_cfg.name == "complex_net",
            "space_name": "observation_space",
        }
        if "central_value_config" in cfg.train.params.config:
            critic_net_cfg = cfg.train.params.config.central_value_config.network
            obs_spec["states"] = {
                "names": list(critic_net_cfg.inputs.keys()),
                "concat": not critic_net_cfg.name == "complex_net",
                "space_name": "state_space",
            }

        vecenv.register(
            "RLGPU",
            lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs),
        )
    else:

        vecenv.register(
            "RLGPU",
            lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs),
        )

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = [RLGPUAlgoObserver()]

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        # Training Agent
        runner.algo_factory.register_builder("hand_continuous", lambda **kwargs: hand_continuous.HandAgent(**kwargs))
        # Player
        runner.player_factory.register_builder(
            "hand_continuous", lambda **kwargs: hand_players.HandPlayerContinuous(**kwargs)
        )
        # Model
        model_builder.register_model(
            "continuous_hand", lambda network, **kwargs: hand_models.ModelCommonContinuous(network)
        )
        # Network
        model_builder.register_network("hand", lambda **kwargs: hand_network_builder.HandBuilder())

        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    if not cfg.test:
        experiment_dir = os.path.join(
            "runs",
            cfg.train.params.config.full_experiment_name,
        )

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    runner.run(
        {
            "train": not cfg.test,
            "play": cfg.test,
            "checkpoint": cfg.checkpoint,
            "sigma": None,
        }
    )


if __name__ == "__main__":
    launch_rlg_hydra()

