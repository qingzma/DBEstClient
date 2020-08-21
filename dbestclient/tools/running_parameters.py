#
# Created by Qingzhi Ma on Tue May 05 2020
#
# Copyright (c) 2020 Department of Computer Science, University of Warwick
# Copyright 2020 Qingzhi Ma
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from copy import deepcopy

from dbestclient.tools.variables import Slaves

RUNTIME_CONF = {
    "device": "cpu",
    "n_jobs": 1,
    'v': True,
    'b_show_latency': True,
    "b_print_to_screen": True,
    "result2file": None,

    # integral related parameters
    "b_use_integral": False,
    "n_division": 20, #20
    # integral package related parameters
    "epsabs": 10.0,
    "epsrel": 0.1,
    "limit": 30,

    "model_suffix": ".dill",
    "slaves": Slaves(),
}


def shrink_runtime_config(runtime_config):
    runtime_config = dict(runtime_config)
    runtime_config.pop("slaves")
    runtime_config.pop("epsabs")
    runtime_config.pop("epsrel")
    runtime_config.pop("limit")
    runtime_config["result2file"] = None
    runtime_config["n_jobs"] = 1
    runtime_config["b_print_to_screen"] = False
    return runtime_config


class DbestConfig:
    """ This is the configuration file for DBEstClient.
    """

    def __init__(self):
        self.config = {
            # system-level configuration.
            # 'warehousedir': '/home/u1796377/Programs/dbestwarehouse',
            'warehousedir': '../dbestwarehouse',
            "reg_type": "mdn",
            "density_type": "mdn",  # qreg
            'backend_server': 'None',
            # "n_jobs": 4,
            "b_grid_search": False,
            # "device": "cpu",
            # "b_reg_mean":'True',
            "b_dummy_gb": False,

            # file format configuration.
            'n_total_point': None,
            'scaling_factor': None,
            'csv_split_char': ',',
            'table_header': None,

            "accept_filter": False,
            # MDN related parameters
            "n_epoch": 20,
            "n_gaussians_reg": 3,
            "n_gaussians_density": 10,
            "b_use_gg": False,
            "n_per_gg": 260,
            "n_hidden_layer": 1,
            "n_mdn_layer_node_reg": 10,
            "n_mdn_layer_node_density": 10,
            "n_embedding_dim":20,
            "encoder": "binary",  # onehot, embedding
            "batch_size": 1000,
            "one_model": False,
        }

    def set_parameters(self, config: dict):
        """ Update the configuration based on a dict.

        Args:
            config (dict): a dictionary with updated values.
        """
        for key in config:
            self.config[key] = config[key]

    def set_parameter(self, key: str, value):
        """ update the configuration.

        Args:
            key (str): the key
            value (str or bool): the value
        """
        self.config[key] = value

    def get_config(self):
        """ Return the configuration for DBEstClient.

        Returns:
            [dict]: the configuration for DBEstClient.
        """
        return self.config

    def copy(self):
        return deepcopy(self)


if __name__ == "__main__":
    conf = DbestConfig()
    print(conf.get_config()["n_per_gg"])
    new_conf = conf.copy()

    conf.set_parameter("n_per_gg", 20)

    print(conf.get_config()["n_per_gg"])
    print(new_conf.get_config()["n_per_gg"])
