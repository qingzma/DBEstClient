#
# Created by Qingzhi Ma on Mon May 04 2020
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

from dbestclient.executor.executor import SqlExecutor


def run():
    config = {
        'warehousedir': '/home/u1796377/Programs/dbestwarehouse',
        'verbose': 'True',
        'b_show_latency': 'True',
        'backend_server': 'None',
        'csv_split_char': ',',
        "epsabs": 10.0,
        "epsrel": 0.1,
        "mesh_grid_num": 20,
        "limit": 30,
        # "b_reg_mean":'True',
        "num_epoch": 400,
        "reg_type": "mdn",
        "density_type": "mdn",
        "num_gaussians": 4,
    }

    sqlExecutor = SqlExecutor()
    # sqlExecutor.set_table_headers("ss_sold_date_sk,ss_sold_time_sk,ss_item_sk,ss_customer_sk,ss_cdemo_sk,ss_hdemo_sk," +
    #                               "ss_addr_sk,ss_store_sk,ss_promo_sk,ss_ticket_number,ss_quantity,ss_wholesale_cost," +
    #                               "ss_list_price,ss_sales_price,ss_ext_discount_amt,ss_ext_sales_price," +
    #                               "ss_ext_wholesale_cost,ss_ext_list_price,ss_ext_tax,ss_coupon_amt,ss_net_paid," +
    #                               "ss_net_paid_inc_tax,ss_net_profit,none")

    build_models(sqlExecutor)
    query(sqlExecutor)


def build_models(sqlExecutor):
    sqlExecutor.execute("create table huawei_test(usermac categorical, ts real,tenantId categorical, ssid  categorical)  "
                        "FROM '/data/huawei/sample.csv' "
                        # "WHERE  ts between 0 and 10 "
                        # "AND tenantId = 'default-organization-id' "
                        # "AND kpiCount = 0 "
                        # "AND ssid = 'Apple' "
                        # "AND regionLevelEight = '9f642594-20c2-4ccb-8f5d-97d5f59a1e18' "
                        "GROUP BY ts "
                        "method uniform "
                        "size 118567 "
                        "scale data;", n_mdn_layer_node=8, encoding="binary", b_grid_search=False, device='gpu', b_use_gg=False, n_per_gg=260)

# SELECT ts, COUNT(DISTINCT usermac)


def query(sqlExecutor):
    sqlExecutor.execute("select count(usermac) from huawei_test "
                        "where ts between 1583402400000 and 1583402400000 "
                        "AND tenantId = 'default-organization-id' "
                        # "AND kpiCount = 0 "
                        "AND ssid = 'Tencent' "
                        # "AND regionLevelEight = '9f642594-20c2-4ccb-8f5d-97d5f59a1e18' "
                        "GROUP BY ts;", n_jobs=1, device='gpu')


if __name__ == "__main__":
    run()