#
# Created by Qingzhi Ma on Thu Jun 04 2020
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


class Query1:
    def __init__(self):
        self.mdl_name = None
        self.sql_executor = None

    def build_model(self, mdl_name: str = "ss_10g", encoder='binary'):
        self.mdl_name = mdl_name
        self.sql_executor = SqlExecutor()

        self.sql_executor.execute("set v='True'")
        # self.sql_executor.execute("set device='cpu'")
        
        self.sql_executor.execute("set b_grid_search='false'")
        self.sql_executor.execute("set b_print_to_screen='false'")
        self.sql_executor.execute("set csv_split_char='|'")
        self.sql_executor.execute("set batch_size=1000")
        self.sql_executor.execute("set table_header=" +
                                  "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
                                  "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
                                  "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
                                  "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
                                  "ss_net_paid_inc_tax|ss_net_profit|none'"
                                  )
        # sql_executor.execute("set table_header=" +
        #                     "'ss_sold_date_sk|ss_store_sk|ss_sales_price'")

        self.sql_executor.execute("set encoder='"+ encoder +"'")
        self.sql_executor.execute("set n_mdn_layer_node_reg=50")          # 5
        self.sql_executor.execute("set n_mdn_layer_node_density=60")      # 30
        self.sql_executor.execute("set n_jobs=1")                         # 2
        self.sql_executor.execute("set n_hidden_layer=2")                 # 1
        self.sql_executor.execute("set n_epoch=20")                       # 20
        self.sql_executor.execute("set n_gaussians_reg=4")                # 3
        self.sql_executor.execute("set n_gaussians_density=20")           # 10

        self.sql_executor.execute(
            "create table "+mdl_name+"(ss_sales_price real, ss_sold_date_sk real) from '../data/tpcds/10g/ss_10g_520k.csv' GROUP BY ss_store_sk method uniform size 1000") # num_points/ss_10g.csv' ")  # num_of_points57.csv

if __name__ == "__main__":
    query = Query1()
    query.build_model(mdl_name='var', encoder='binary')
    # query.sql_executor.execute("select ss_store_sk, var(y) from var group by ss_store_sk")
    query.sql_executor.execute("select ss_store_sk, var(ss_sold_date_sk) from ss_10g_binary group by ss_store_sk")