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
            "create table "+mdl_name+"(ss_sales_price real, ss_sold_date_sk real) from '../data/tpcds/10g/ss_10g_520k.csv' GROUP BY ss_store_sk method uniform size num_points/ss_10g.csv' ")  # num_of_points57.csv
        
        # self.sql_executor.execute(
        #     "create table "+"ss10g_binary_30"+"(ss_sales_price real, ss_sold_date_sk real) from '../data/tpcds/10g/ss_10g_520k.csv' GROUP BY ss_store_sk method uniform size 'num_points/ss_10g.csv' ")  # num_of_points57.csv


    def workload(self, mdl_name, result2file: str = '/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn/10g/', n_jobs=1):
        self.sql_executor.mdl_name = mdl_name
        self.sql_executor.execute("set n_jobs=" + str(n_jobs)+'"')
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum1.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                  "  where   2451119 <=ss_sold_date_sk<= 2451483 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum2.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451300 <=ss_sold_date_sk<= 2451665 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum3.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451392 <=ss_sold_date_sk<= 2451757 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum4.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451484 <=ss_sold_date_sk<= 2451849 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum5.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451545 <=ss_sold_date_sk<= 2451910 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum6.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451636 <=ss_sold_date_sk<= 2452000 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum7.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451727 <=ss_sold_date_sk<= 2452091 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum8.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451850 <=ss_sold_date_sk<= 2452214 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum9.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451911 <=ss_sold_date_sk<= 2452275 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum10.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2452031 <=ss_sold_date_sk<= 2452395 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count1.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                  "  where  2451119 <=ss_sold_date_sk<= 2451483 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count2.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                  "  where  2451300 <=ss_sold_date_sk<= 2451665 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count3.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                  "  where  2451392 <=ss_sold_date_sk<= 2451757 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count4.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                  "  where  2451484 <=ss_sold_date_sk<= 2451849 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count5.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                  "  where  2451545 <=ss_sold_date_sk<= 2451910 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count6.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                  "  where  2451636 <=ss_sold_date_sk<= 2452000 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count7.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                  "  where  2451727 <=ss_sold_date_sk<= 2452091 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count8.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                  "  where  2451850 <=ss_sold_date_sk<= 2452214 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count9.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                  "  where  2451911 <=ss_sold_date_sk<= 2452275 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count10.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                  "  where  2452031 <=ss_sold_date_sk<= 2452395 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg1.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451119 <=ss_sold_date_sk<= 2451483 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg2.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451300 <=ss_sold_date_sk<= 2451665 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg3.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451392 <=ss_sold_date_sk<= 2451757 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg4.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451484 <=ss_sold_date_sk<= 2451849 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg5.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451545 <=ss_sold_date_sk<= 2451910 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg6.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451636 <=ss_sold_date_sk<= 2452000 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg7.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451727 <=ss_sold_date_sk<= 2452091 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg8.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451850 <=ss_sold_date_sk<= 2452214 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg9.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2451911 <=ss_sold_date_sk<= 2452275 group by   ss_store_sk",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg10.txt'")
        # self.sql_executor.execute(
            # "set result2file='None'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                  "  where  2452031 <=ss_sold_date_sk<= 2452395 group by   ss_store_sk",)


if __name__ == "__main__":
    query1 = Query1()
    query1.build_model(mdl_name="ss_10g_binary",encoder="binary")
    query1.build_model(mdl_name="ss_10g_onehot",encoder="onehot")
    query1.build_model(mdl_name="ss_10g_embedding",encoder="embedding")
    query1.workload("ss10g_embedding_30",result2file="experiments/results/mdn/10g/")
