# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Configurations for HiFiC."""

from . import helpers


_CONFIGS = {
    'hific': helpers.Config(
        model_type=helpers.ModelType.COMPRESSION_GAN,
        lambda_schedule=helpers.Config(
            vals=[2., 1.],
            steps=[50000]),
        lr=1e-4,
        lr_schedule=helpers.Config(
            vals=[1., 0.1],
            steps=[500000]),
        num_steps_disc=1,
        loss_config=helpers.Config(
            # Constrain rate:
            #   Loss = C * (1/lambda * R + CD * D) + CP * P
            #       where
            #          lambda = lambda_a if current_bpp > target
            #                   lambda_b otherwise.
            CP=0.1 * 1.5 ** 1,  # Sweep over 0.1 * 1.5 ** x
            C=0.1 * 2. ** -5,
            CD=0.75,
            target=0.14,  # This is $r_t$ in the paper.
            lpips_weight=1.,
            target_schedule=helpers.Config(
                vals=[0.20/0.14, 1.],  # Factor is independent of target.
                steps=[50000]),
            lmbda_a=0.1 * 2. ** -6,
            lmbda_b=0.1 * 2. ** 1,
            )
        ),
    'mselpips': helpers.Config(
        model_type=helpers.ModelType.COMPRESSION,
        lambda_schedule=helpers.Config(
            vals=[2., 1.],
            steps=[50000]),
        lr=1e-4,
        lr_schedule=helpers.Config(
            vals=[1., 0.1],
            steps=[500000]),
        num_steps_disc=None,
        loss_config=helpers.Config(
            # Constrain rate:
            #   Loss = C * (1/lambda * R + CD * D) + CP * P
            #       where
            #          lambda = lambda_a if current_bpp > target
            #                   lambda_b otherwise.
            CP=None,
            C=0.1 * 2. ** -5,
            CD=0.75,
            target=0.14,  # This is $r_t$ in the paper.
            lpips_weight=1.,
            target_schedule=helpers.Config(
                vals=[0.20/0.14, 1.],  # Factor is independent of target.
                steps=[50000]),
            lmbda_a=0.1 * 2. ** -6,
            lmbda_b=0.1 * 2. ** 1,
            )
        )
}


def get_config(config_name):
  if config_name not in _CONFIGS:
    raise ValueError(f'Unknown config_name={config_name} not in '
                     f'{_CONFIGS.keys()}')
  return _CONFIGS[config_name]


def valid_configs():
  return list(_CONFIGS.keys())

