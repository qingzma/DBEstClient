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
# SELECT unique_carrier, COUNT(dep_delay) FROM flights GROUP BY unique_carrier;
# hive -e "SELECT unique_carrier, COUNT(*) FROM flights GROUP BY unique_carrier" > flights_group.csv

# hive -e "SELECT unique_carrier, AVG(dep_delay) FROM flights WHERE distance >=  300  AND distance <= 1000 GROUP BY unique_carrier" > flights_avg1.csv
# hive -e "SELECT unique_carrier, AVG(dep_delay) FROM flights WHERE distance >= 1000  AND distance <= 1500 GROUP BY unique_carrier" > flights_avg2.csv
# hive -e "SELECT unique_carrier, AVG(dep_delay) FROM flights WHERE distance >= 1500  AND distance <= 2000 GROUP BY unique_carrier" > flights_avg3.csv
# hive -e "SELECT unique_carrier, SUM(dep_delay) FROM flights WHERE distance >=  300  AND distance <= 1000 GROUP BY unique_carrier" > flights_sum1.csv
# hive -e "SELECT unique_carrier, SUM(dep_delay) FROM flights WHERE distance >= 1000  AND distance <= 1500 GROUP BY unique_carrier" > flights_sum2.csv
# hive -e "SELECT unique_carrier, SUM(dep_delay) FROM flights WHERE distance >= 1500  AND distance <= 2000 GROUP BY unique_carrier" > flights_sum3.csv

# hive -e "SELECT unique_carrier, COUNT(*) FROM flights WHERE dep_delay>=1000 AND dep_delay<=1200 AND origin_state_abr='LA'  GROUP BY unique_carrier" > flights_one_model_1_x.csv

from dbestclient.executor.executor import SqlExecutor


class Query1:
    def __init__(self):
        self.mdl_name = None
        self.sql_executor = None
        self.model_size = "1m"

    def build_model(self, mdl_name: str = "flight_1m", encoder='binary'):
        self.mdl_name = mdl_name
        self.sql_executor = SqlExecutor()

        self.sql_executor.execute("set v='True'")
        # self.sql_executor.execute("set device='cpu'")
        
        self.sql_executor.execute("set b_grid_search='false'")
        self.sql_executor.execute("set b_print_to_screen='false'")
        self.sql_executor.execute("set csv_split_char=','")
        self.sql_executor.execute("set batch_size=1000")
        self.sql_executor.execute("set table_header=" +
                                  "'year_date,unique_carrier,origin,origin_state_abr,dest,dest_state_abr,dep_delay,taxi_out,taxi_in,arr_delay,air_time,distance'")
        
        self.sql_executor.execute("set encoder='"+ encoder +"'")
        
        if self.model_size =="1m":
            self.sql_executor.execute("set n_mdn_layer_node_reg=10")          # 20
            self.sql_executor.execute("set n_mdn_layer_node_density=15")      # 30
            self.sql_executor.execute("set n_jobs=1")                         # 
            self.sql_executor.execute("set n_hidden_layer=1")                 # 2
            self.sql_executor.execute("set n_epoch=20")                       # 50
            self.sql_executor.execute("set n_gaussians_reg=8")                # 
            self.sql_executor.execute("set n_gaussians_density=10")           # 8
            self.sql_executor.execute(
                "create table "+mdl_name+"(dep_delay real, distance real) from '../data/flights/flight_1m.csv' GROUP BY unique_carrier method uniform size 'num_points/flights_group.csv' ") 
        elif self.model_size == "5m":
            self.sql_executor.execute("set n_mdn_layer_node_reg=20")          # 20
            self.sql_executor.execute("set n_mdn_layer_node_density=30")      # 30
            self.sql_executor.execute("set n_jobs=1")                         # 
            self.sql_executor.execute("set n_hidden_layer=1")                 # 1
            self.sql_executor.execute("set n_epoch=20")                       # 20
            self.sql_executor.execute("set n_gaussians_reg=8")                # 
            self.sql_executor.execute("set n_gaussians_density=8")            # 10
            self.sql_executor.execute(
                "create table "+mdl_name+"(dep_delay real, distance real) from '../data/flights/flight_5m.csv' GROUP BY unique_carrier method uniform size 'num_points/flights_group.csv' ") 
        

        # self.sql_executor.execute("set n_mdn_layer_node_reg=30")          # 5
        # self.sql_executor.execute("set n_mdn_layer_node_density=30")      # 30
        # self.sql_executor.execute("set n_jobs=1")                         # 2
        # self.sql_executor.execute("set n_hidden_layer=1")                 # 1
        # self.sql_executor.execute("set n_epoch=20")                       # 20
        # self.sql_executor.execute("set n_gaussians_reg=8")                # 3
        # self.sql_executor.execute("set n_gaussians_density=8")           # 10

        # self.sql_executor.execute(
        #     "create table "+mdl_name+"(dep_delay real, distance real) from '../data/flights/flights_10k.csv' GROUP BY unique_carrier method uniform size 'num_points/flights_group.csv' ")  # num_of_points57.csv
        
        # SELECT unique_carrier, AVG(dep_delay) FROM flights WHERE 300<=distance<=1000 GROUP BY unique_carrier;
        # self.sql_executor.execute(
        #     "create table "+"ss10g_binary_30"+"(ss_sales_price real, ss_sold_date_sk real) from '../data/tpcds/10g/ss_10g_520k.csv' GROUP BY ss_store_sk method uniform size 'num_points/ss_10g.csv' ")  # num_of_points57.csv

    def build_one_model(self, mdl_name: str = "flight_1m", encoder='binary'):
        self.mdl_name = mdl_name
        self.sql_executor = SqlExecutor()

        self.sql_executor.execute("set v='True'")
        # self.sql_executor.execute("set device='cpu'")
        self.sql_executor.execute("set one_model='true'")
        self.sql_executor.execute("set b_grid_search='false'")
        self.sql_executor.execute("set b_print_to_screen='false'")
        self.sql_executor.execute("set csv_split_char=','")
        self.sql_executor.execute("set batch_size=1000")
        self.sql_executor.execute("set table_header=" +
                                  "'year_date,unique_carrier,origin,origin_state_abr,dest,dest_state_abr,dep_delay,taxi_out,taxi_in,arr_delay,air_time,distance'")
        
        self.sql_executor.execute("set encoder='"+ encoder +"'")

        model_size = "5m"
        if model_size =="1m":
            self.sql_executor.execute("set n_mdn_layer_node_reg=20")          # 5
            self.sql_executor.execute("set n_mdn_layer_node_density=30")      # 30
            self.sql_executor.execute("set n_jobs=1")                         # 2
            self.sql_executor.execute("set n_hidden_layer=2")                 # 1
            self.sql_executor.execute("set n_epoch=20")                       # 20
            self.sql_executor.execute("set n_gaussians_reg=8")                # 3
            self.sql_executor.execute("set n_gaussians_density=8")            # 10
            self.sql_executor.execute(
            "create table "+mdl_name+"(distance real, dep_delay real, origin_state_abr categorical) from '../data/flights/flight_1m.csv' GROUP BY unique_carrier method uniform size 0.001")
        elif model_size == "5m":
            self.sql_executor.execute("set n_mdn_layer_node_reg=50")          # 20
            self.sql_executor.execute("set n_mdn_layer_node_density=50")      # 30
            self.sql_executor.execute("set n_jobs=1")                         # 
            self.sql_executor.execute("set n_hidden_layer=2")                 # 1
            self.sql_executor.execute("set n_epoch=20")                       # 50
            self.sql_executor.execute("set n_gaussians_reg=8")                # 3
            self.sql_executor.execute("set n_gaussians_density=8")            # 10
            self.sql_executor.execute(
            "create table "+mdl_name+"(distance real, dep_delay real, origin_state_abr categorical) from '../data/flights/flight_5m.csv' GROUP BY unique_carrier method uniform size 0.005")

        # self.sql_executor.execute(
        #     "create table "+mdl_name+"(distance real, dep_delay real, origin_state_abr categorical) from '../data/flights/flight_5m.csv' GROUP BY unique_carrier method uniform size 0.005")#'num_points/flights_group.csv' ")  # num_of_points57.csv
        #SELECT unique_carrier, COUNT(*) FROM flights WHERE origin_state_abr='LA' AND  dest_state_abr='CA' GROUP BY unique_carrier;

    def workload(self, mdl_name, result2file: str = 'experiments/flights/results/mdn1m/', n_jobs=1):
        self.sql_executor.mdl_name = mdl_name
        self.sql_executor.execute("set n_jobs=" + str(n_jobs)+'"')
        # self.sql_executor.execute(
        #     "set result2file='" + result2file + "7.txt'")
        # self.sql_executor.execute("select unique_carrier, COUNT(dep_delay) from  " + self.mdl_name +
        #                           "  where   300<=distance<=1000 GROUP BY unique_carrier")
        # self.sql_executor.execute(
        #     "set result2file='" + result2file + "8.txt'")
        # self.sql_executor.execute("SELECT unique_carrier, COUNT(dep_delay) from " + self.mdl_name +
        #                           "  where   1000<=distance<=1500 GROUP BY unique_carrier",)
        # self.sql_executor.execute(
        #     "set result2file='" + result2file + "9.txt'")
        # self.sql_executor.execute("SELECT unique_carrier, COUNT(dep_delay) from " + self.mdl_name +
        #                           "  where   1500<=distance<=2000 GROUP BY unique_carrier",)
        # self.sql_executor.execute(
        #     "set result2file='" + result2file + "10.txt'")
        # self.sql_executor.execute("SELECT unique_carrier, SUM(dep_delay) from " + self.mdl_name +
        #                           "  where   300<=distance<=1000 GROUP BY unique_carrier",)
        # self.sql_executor.execute(
        #     "set result2file='" + result2file + "11.txt'")
        # self.sql_executor.execute("SELECT unique_carrier, SUM(dep_delay) from " + self.mdl_name +
        #                           "  where   1000<=distance<=1500 GROUP BY unique_carrier",)
        # self.sql_executor.execute(
        #     "set result2file='" + result2file + "12.txt'")
        # self.sql_executor.execute("SELECT unique_carrier, SUM(dep_delay) from " + self.mdl_name +
        #                           "  where   1500<=distance<=2000 GROUP BY unique_carrier",)
        # self.sql_executor.execute(
        #     "set result2file='" + result2file + "13.txt'")
        # self.sql_executor.execute("SELECT unique_carrier, AVG(dep_delay) from " + self.mdl_name +
        #                           "  where   300<=distance<=1000 GROUP BY unique_carrier",)
        # self.sql_executor.execute(
        #     "set result2file='" + result2file + "14.txt'")
        # self.sql_executor.execute("SELECT unique_carrier, AVG(dep_delay) from " + self.mdl_name +
        #                           "  where   1000<=distance<=1500 GROUP BY unique_carrier",)
        # self.sql_executor.execute(
        #     "set result2file='" + result2file + "15.txt'")
        # self.sql_executor.execute("SELECT unique_carrier, AVG(dep_delay) from " + self.mdl_name +
        #                           "  where   1500<=distance<=2000 GROUP BY unique_carrier",)
        
        self.sql_executor.execute(
            "set result2file='" + result2file + "one_model_1x.txt'")
        
        self.sql_executor.execute("SELECT unique_carrier, COUNT(distance) FROM "+mdl_name+" where 1000<=dep_delay<=1200 AND origin_state_abr='LA'  GROUP BY unique_carrier")
        


if __name__ == "__main__":
    query1 = Query1()
    query1.model_size = "1m"
    query1.build_model(mdl_name="flights_1m_binary_small",encoder="binary")
    # # query1.build_model(mdl_name="flights_1m_onehot",encoder="onehot")
    # # query1.build_model(mdl_name="flights_1m_embedding",encoder="embedding")
    # query1.workload("flights_5m_binary",result2file="experiments/flights/results/mdn5m/")

    query1.build_one_model("flight_one_model1",encoder="binary")
    # query1.build_one_model("flight_one_model_embedding",encoder="embedding")
    query1.workload("flight_one_model1",result2file="experiments/flights/results/mdn5m/")


