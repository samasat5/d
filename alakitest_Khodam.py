
# transition_dim = 10
# dim=32
# dim_mults=(1, 2, 4, 8)
# dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
# in_out = list(zip(dims[:-1], dims[1:]))

# for ind, (dim_in, dim_out) in enumerate(in_out):
#     print(ind, (dim_in, dim_out))
# print (dims)
# print (in_out)
# for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
#     print(ind, (dim_in, dim_out))
# print ((in_out[1:]))

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.replay_buffer_nosave import ReplayBuffer2
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
import torch
import numpy as np
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.blockpush_lowdim_dataset import BlockPushLowdimDataset
from diffusion_policy.dataset.blockpush_lowdim_dataset2 import BlockPushLowdimDataset2
from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset
from omegaconf import OmegaConf
from diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace2 import TrainDiffusionTransformerLowdimWorkspace2
import hydra
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
import zarr
from zarr.storage import LocalStore

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import tqdm
import numpy as np
import shutil

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

zarr_path_root = "./block_pushing/multimodal_push_seed.zarr/data"
root = zarr.open(zarr_path_root, mode='r')

print(root.tree())  # Show group structure


zarr_path = "./block_pushing/multimodal_push_seed.zarr"
buffer = ReplayBuffer2.copy_from_path(zarr_path, backend='numpy')
print(buffer.backend)         # 'numpy'
print(buffer.n_episodes)      # number of episodes
print(buffer.n_steps)         # total steps
print(buffer.keys())          # e.g., ['obs', 'action']


# Inspect buffer
print("Keys:", buffer.keys())
print("Number of episodes:", buffer.n_episodes)
print("Number of steps:", buffer.n_steps)

# Access first episode
episode0 = buffer.get_episode(0)

print("Episode 0 keys:", episode0.keys())
print("Episode 0 obs shape:", episode0['obs'].shape)
print("Episode 0 action shape:", episode0['action'].shape)

# Access a slice of steps (e.g., steps 10 to 20)
slice_data = buffer.get_steps_slice(0,104)
print("Slice obs shape:", slice_data['obs'].shape)


sampler = SequenceSampler(
            replay_buffer=buffer, 
            sequence_length=1,
            pad_before=1, 
            pad_after=1)

sample0 = sampler.sample_sequence(0)

# bs_eef_target= True
def _sample_to_data(sample):
    obs = sample0['obs'] # T, D_o
    data = {
        'obs': obs,
        'action': sample0['action'], # T, D_a
    }
    return data

data = _sample_to_data(sample0)
torch_data = dict_apply(data, torch.from_numpy)
print("torch", torch_data)
# print("obs shape - sample of ep0:", sample0['obs'].shape)
# print(f"action shape of ep 0:\n{episode0['action'][:4]}")
# print("action shape - sample0 of ep0:\n", sample0['action'])
# print(f"obs shape of ep 0:{episode0['obs'].shape}")
# print("\nFirst 3 obs rows:\n", sample['obs'][:3])
# print("\nFirst 3 action rows:\n", sample['action'][:3])

# Create dataset
dataset = BlockPushLowdimDataset2(
    zarr_path=zarr_path,
    obs_key='obs',        # Zarr dataset must have 'state' key
    action_key='action',    # Zarr dataset must have 'action' key
         # Reserve 10% of episodes for validation
)

# # Print one sample
# sample = dataset[0]
# print(dataset[2])
# print("Sample keys:", sample)
# print("Sample keys:", sample.keys())
# print("Obs shape:", sample['obs'].shape)
# print("Action shape:", sample['action'].shape)

# # Load a replay buffer from disk
# rb = ReplayBuffer.copy_from_path('D:/MUSIC/Apppp/Sorbonne 2nd yr/Reproduce/pusht/pusht_cchi_v7_replay.zarr',
#                                  keys=['state', 'action'])

# # Add a new episode
# rb.add_episode({
#     'state': np.random.rand(10, 4),
#     'action': np.random.rand(10, 2)
# })

# # Retrieve one episode
# ep = rb.get_episode(0)

# # Save to a new path
# rb.save_to_path('D:/MUSIC/Apppp/Sorbonne 2nd yr/Reproduce')


# # Load the YAML config file
# cfg = OmegaConf.load("diffusion_policy/config/train_diffusion_transformer_lowdim_workspace2.yaml")

# # Instantiate and run training
# workspace = TrainDiffusionTransformerLowdimWorkspace2(cfg)
# workspace.run()
# from omegaconf import DictConfig

# @hydra.main(config_path="diffusion_policy/config", config_name="train_diffusion_transformer_lowdim_workspace2",version_base=None)
# def main(cfg: DictConfig):
#     workspace = TrainDiffusionTransformerLowdimWorkspace2(cfg)
#     workspace.run()

# if __name__ == "__main__":
# =======
# transition_dim = 10
# dim=32
# dim_mults=(1, 2, 4, 8)
# dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
# in_out = list(zip(dims[:-1], dims[1:]))

# for ind, (dim_in, dim_out) in enumerate(in_out):
#     print(ind, (dim_in, dim_out))
# print (dims)
# print (in_out)
# for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
#     print(ind, (dim_in, dim_out))
# print ((in_out[1:]))

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.replay_buffer_nosave import ReplayBuffer2
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
import torch
import numpy as np
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.blockpush_lowdim_dataset import BlockPushLowdimDataset
from diffusion_policy.dataset.blockpush_lowdim_dataset2 import BlockPushLowdimDataset2
from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset
from omegaconf import OmegaConf
from diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace2 import TrainDiffusionTransformerLowdimWorkspace2
import hydra
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
import zarr
from zarr.storage import LocalStore

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import tqdm
import numpy as np
import shutil

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

zarr_path_root = "./block_pushing/multimodal_push_seed.zarr/data"
root = zarr.open(zarr_path_root, mode='r')

print(root.tree())  # Show group structure


zarr_path = "./block_pushing/multimodal_push_seed.zarr"
buffer = ReplayBuffer2.copy_from_path(zarr_path, backend='numpy')
print(buffer.backend)         # 'numpy'
print(buffer.n_episodes)      # number of episodes
print(buffer.n_steps)         # total steps
print(buffer.keys())          # e.g., ['obs', 'action']


# Inspect buffer
print("Keys:", buffer.keys())
print("Number of episodes:", buffer.n_episodes)
print("Number of steps:", buffer.n_steps)

# Access first episode
episode0 = buffer.get_episode(0)

print("Episode 0 keys:", episode0.keys())
print("Episode 0 obs shape:", episode0['obs'].shape)
print("Episode 0 action shape:", episode0['action'].shape)

# Access a slice of steps (e.g., steps 10 to 20)
slice_data = buffer.get_steps_slice(0,104)
print("Slice obs shape:", slice_data['obs'].shape)


sampler = SequenceSampler(
            replay_buffer=buffer, 
            sequence_length=1,
            pad_before=1, 
            pad_after=1)

sample0 = sampler.sample_sequence(0)

# bs_eef_target= True
def _sample_to_data(sample):
    obs = sample0['obs'] # T, D_o
    data = {
        'obs': obs,
        'action': sample0['action'], # T, D_a
    }
    return data

data = _sample_to_data(sample0)
torch_data = dict_apply(data, torch.from_numpy)
print("torch", torch_data)
# print("obs shape - sample of ep0:", sample0['obs'].shape)
# print(f"action shape of ep 0:\n{episode0['action'][:4]}")
# print("action shape - sample0 of ep0:\n", sample0['action'])
# print(f"obs shape of ep 0:{episode0['obs'].shape}")
# print("\nFirst 3 obs rows:\n", sample['obs'][:3])
# print("\nFirst 3 action rows:\n", sample['action'][:3])

# Create dataset
dataset = BlockPushLowdimDataset2(
    zarr_path=zarr_path,
    obs_key='obs',        # Zarr dataset must have 'state' key
    action_key='action',    # Zarr dataset must have 'action' key
         # Reserve 10% of episodes for validation
)

# # Print one sample
# sample = dataset[0]
# print(dataset[2])
# print("Sample keys:", sample)
# print("Sample keys:", sample.keys())
# print("Obs shape:", sample['obs'].shape)
# print("Action shape:", sample['action'].shape)

# # Load a replay buffer from disk
# rb = ReplayBuffer.copy_from_path('D:/MUSIC/Apppp/Sorbonne 2nd yr/Reproduce/pusht/pusht_cchi_v7_replay.zarr',
#                                  keys=['state', 'action'])

# # Add a new episode
# rb.add_episode({
#     'state': np.random.rand(10, 4),
#     'action': np.random.rand(10, 2)
# })

# # Retrieve one episode
# ep = rb.get_episode(0)

# # Save to a new path
# rb.save_to_path('D:/MUSIC/Apppp/Sorbonne 2nd yr/Reproduce')


# # Load the YAML config file
# cfg = OmegaConf.load("diffusion_policy/config/train_diffusion_transformer_lowdim_workspace2.yaml")

# # Instantiate and run training
# workspace = TrainDiffusionTransformerLowdimWorkspace2(cfg)
# workspace.run()
# from omegaconf import DictConfig

# @hydra.main(config_path="diffusion_policy/config", config_name="train_diffusion_transformer_lowdim_workspace2",version_base=None)
# def main(cfg: DictConfig):
#     workspace = TrainDiffusionTransformerLowdimWorkspace2(cfg)
#     workspace.run()

# if __name__ == "__main__":
#     main()