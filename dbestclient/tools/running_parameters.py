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


class DbestConfig:
    """ This is the configuration file for DBEstClient.
    """

    def __init__(self):
        self.config = {
            # system-level configuration.
            'warehousedir': '/home/u1796377/Programs/dbestwarehouse',
            'verbose': True,
            'b_show_latency': True,
            "b_print_to_screen": True,
            "reg_type": "mdn",
            "density_type": "mdn",
            'backend_server': 'None',
            "n_jobs": 4,
            "b_grid_search": True,
            "device": "cpu",
            # "b_reg_mean":'True',

            # file format configuration.
            'csv_split_char': ',',

            # MDN related parameters
            "num_epoch": 400,
            "num_gaussians": 4,
            "b_use_gg": False,
            "n_per_gg": 10,
            "result2file": None,
            "n_mdn_layer_node": 10,
            "encoding": "binary",  # one-hot

            # integral related parameters
            "b_use_integral": False,
            "n_division": 20,
            # integral package related parameters
            "epsabs": 10.0,
            "epsrel": 0.1,
            "limit": 30,
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


# if __name__ == "__main__":
#     conf = DbestConfig()
#     conf.update({})
