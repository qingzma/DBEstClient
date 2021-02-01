
# hive -e "select ss_store_sk,ss_quantity, count(*) from store_sales_1t  group by ss_store_sk,ss_quantity;" > ~/large_group_counts.csv
from dbestclient.executor.executor import SqlExecutor
class Query1:
    def __init__(self):
        self.mdl_name = None
        self.sql_executor = None

    def build_model2_5m(self, mdl_name: str = "ss_gb2_2_5", encoder='embedding'):
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
        # self.sql_executor.execute("set n_mdn_layer_node_reg=50")          # 5
        # self.sql_executor.execute("set n_mdn_layer_node_density=60")      # 30
        self.sql_executor.execute("set n_jobs=1")                         # 2
        # self.sql_executor.execute("set n_hidden_layer=2")                 # 1
        # self.sql_executor.execute("set n_epoch=20")                       # 20
        # self.sql_executor.execute("set n_gaussians_reg=4")                # 3
        # self.sql_executor.execute("set n_gaussians_density=20")           # 10

        self.sql_executor.execute("set n_mdn_layer_node_reg=50")          
        self.sql_executor.execute("set n_mdn_layer_node_density=60")      
        self.sql_executor.execute("set n_hidden_layer=1")      
        self.sql_executor.execute("set n_epoch=30")                   
        self.sql_executor.execute("set n_gaussians_reg=8")                
        self.sql_executor.execute("set n_gaussians_density=12")            
        self.sql_executor.execute("set n_embedding_dim=20") 

        self.sql_executor.execute(
            "create table "+mdl_name+"(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/store_sales.dat' GROUP BY ss_store_sk,ss_quantity method stratified size 60' ")  # num_of_points57.csv

    def build_model5_2m(self, mdl_name: str = "ss_gb2_5_2", encoder='embedding'):
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
        # self.sql_executor.execute("set n_mdn_layer_node_reg=50")          # 5
        # self.sql_executor.execute("set n_mdn_layer_node_density=60")      # 30
        self.sql_executor.execute("set n_jobs=1")                         # 2
        # self.sql_executor.execute("set n_hidden_layer=2")                 # 1
        # self.sql_executor.execute("set n_epoch=20")                       # 20
        # self.sql_executor.execute("set n_gaussians_reg=4")                # 3
        # self.sql_executor.execute("set n_gaussians_density=20")           # 10

        self.sql_executor.execute("set n_mdn_layer_node_reg=20")          
        self.sql_executor.execute("set n_mdn_layer_node_density=30")      
        self.sql_executor.execute("set n_hidden_layer=1")      
        self.sql_executor.execute("set n_epoch=20")                   
        self.sql_executor.execute("set n_gaussians_reg=8")                
        self.sql_executor.execute("set n_gaussians_density=8")            
        self.sql_executor.execute("set n_embedding_dim=10") 

        self.sql_executor.execute(
            "create table "+mdl_name+"(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/store_sales.dat' GROUP BY ss_store_sk,ss_quantity method stratified size 114' ")  # num_of_points57.csv
    
    def build_model_10m(self, mdl_name: str = "ss_gb2_5_2", encoder='embedding'):
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
        # self.sql_executor.execute("set n_mdn_layer_node_reg=50")          # 5
        # self.sql_executor.execute("set n_mdn_layer_node_density=60")      # 30
        self.sql_executor.execute("set n_jobs=1")                         # 2
        # self.sql_executor.execute("set n_hidden_layer=2")                 # 1
        # self.sql_executor.execute("set n_epoch=20")                       # 20
        # self.sql_executor.execute("set n_gaussians_reg=4")                # 3
        # self.sql_executor.execute("set n_gaussians_density=20")           # 10

        self.sql_executor.execute("set n_mdn_layer_node_reg=20")          
        self.sql_executor.execute("set n_mdn_layer_node_density=30")      
        self.sql_executor.execute("set n_hidden_layer=1")      
        self.sql_executor.execute("set n_epoch=20")                   
        self.sql_executor.execute("set n_gaussians_reg=8")                
        self.sql_executor.execute("set n_gaussians_density=8")            
        self.sql_executor.execute("set n_embedding_dim=10") 

        self.sql_executor.execute(
            "create table "+mdl_name+"(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/store_sales.dat' GROUP BY ss_store_sk,ss_quantity method stratified size 220' ")  # num_of_points57.csv

    
    def build_model_20m(self, mdl_name: str = "ss_gb2_5_2", encoder='embedding'):
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
        # self.sql_executor.execute("set n_mdn_layer_node_reg=50")          # 5
        # self.sql_executor.execute("set n_mdn_layer_node_density=60")      # 30
        self.sql_executor.execute("set n_jobs=1")                         # 2
        # self.sql_executor.execute("set n_hidden_layer=2")                 # 1
        # self.sql_executor.execute("set n_epoch=20")                       # 20
        # self.sql_executor.execute("set n_gaussians_reg=4")                # 3
        # self.sql_executor.execute("set n_gaussians_density=20")           # 10

        self.sql_executor.execute("set n_mdn_layer_node_reg=20")          
        self.sql_executor.execute("set n_mdn_layer_node_density=30")      
        self.sql_executor.execute("set n_hidden_layer=1")      
        self.sql_executor.execute("set n_epoch=20")                   
        self.sql_executor.execute("set n_gaussians_reg=8")                
        self.sql_executor.execute("set n_gaussians_density=8")            
        self.sql_executor.execute("set n_embedding_dim=10") 

        self.sql_executor.execute(
            "create table "+mdl_name+"(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/store_sales.dat' GROUP BY ss_store_sk,ss_quantity method stratified size 430' ")  # num_of_points57.csv

        
    
    def build_model_30m(self, mdl_name: str = "ss_gb2_5_2", encoder='embedding'):
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
        # self.sql_executor.execute("set n_mdn_layer_node_reg=50")          # 5
        # self.sql_executor.execute("set n_mdn_layer_node_density=60")      # 30
        self.sql_executor.execute("set n_jobs=1")                         # 2
        # self.sql_executor.execute("set n_hidden_layer=2")                 # 1
        # self.sql_executor.execute("set n_epoch=20")                       # 20
        # self.sql_executor.execute("set n_gaussians_reg=4")                # 3
        # self.sql_executor.execute("set n_gaussians_density=20")           # 10

        self.sql_executor.execute("set n_mdn_layer_node_reg=20")          
        self.sql_executor.execute("set n_mdn_layer_node_density=30")      
        self.sql_executor.execute("set n_hidden_layer=1")      
        self.sql_executor.execute("set n_epoch=20")                   
        self.sql_executor.execute("set n_gaussians_reg=8")                
        self.sql_executor.execute("set n_gaussians_density=8")            
        self.sql_executor.execute("set n_embedding_dim=10") 

        self.sql_executor.execute(
            "create table "+mdl_name+"(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/store_sales.dat' GROUP BY ss_store_sk,ss_quantity method stratified size 640' ")  # num_of_points57.csv
    
    def query_workload(self, mdl_name, result2file: str = '/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn/10g/', n_jobs=1):
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum1.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                "  where   2451119 <=ss_sold_date_sk<= 2451483 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum2.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451300 <=ss_sold_date_sk<= 2451665 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum3.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451392 <=ss_sold_date_sk<= 2451757 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum4.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451484 <=ss_sold_date_sk<= 2451849 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum5.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451545 <=ss_sold_date_sk<= 2451910 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum6.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451636 <=ss_sold_date_sk<= 2452000 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum7.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451727 <=ss_sold_date_sk<= 2452091 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum8.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451850 <=ss_sold_date_sk<= 2452214 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum9.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451911 <=ss_sold_date_sk<= 2452275 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "sum10.txt'")
        self.sql_executor.execute("select sum(ss_sales_price)   from " + self.mdl_name +
                                "  where  2452031 <=ss_sold_date_sk<= 2452395 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count1.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                "  where  2451119 <=ss_sold_date_sk<= 2451483 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count2.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                "  where  2451300 <=ss_sold_date_sk<= 2451665 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count3.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                "  where  2451392 <=ss_sold_date_sk<= 2451757 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count4.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                "  where  2451484 <=ss_sold_date_sk<= 2451849 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count5.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                "  where  2451545 <=ss_sold_date_sk<= 2451910 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count6.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                "  where  2451636 <=ss_sold_date_sk<= 2452000 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count7.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                "  where  2451727 <=ss_sold_date_sk<= 2452091 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count8.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                "  where  2451850 <=ss_sold_date_sk<= 2452214 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count9.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                "  where  2451911 <=ss_sold_date_sk<= 2452275 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "count10.txt'")
        self.sql_executor.execute("select count(ss_sales_price) from " + self.mdl_name +
                                "  where  2452031 <=ss_sold_date_sk<= 2452395 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg1.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451119 <=ss_sold_date_sk<= 2451483 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg2.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451300 <=ss_sold_date_sk<= 2451665 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg3.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451392 <=ss_sold_date_sk<= 2451757 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg4.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451484 <=ss_sold_date_sk<= 2451849 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg5.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451545 <=ss_sold_date_sk<= 2451910 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg6.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451636 <=ss_sold_date_sk<= 2452000 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg7.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451727 <=ss_sold_date_sk<= 2452091 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg8.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451850 <=ss_sold_date_sk<= 2452214 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg9.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                "  where  2451911 <=ss_sold_date_sk<= 2452275 group by   ss_store_sk,ss_quantity",)
        self.sql_executor.execute(
            "set result2file='" + result2file + "avg10.txt'")
        self.sql_executor.execute("select avg(ss_sales_price)   from " + self.mdl_name +
                                "  where  2452031 <=ss_sold_date_sk<= 2452395 group by   ss_store_sk,ss_quantity",)


if  __name__ == "__main__":
    q1= Query1()
    # q1.build_model2_5m("ss_gb2_2_5")
    # q1.mdl_name="ss_gb2_2_5"
    # q1.query_workload("ss_gb2_2_5",result2file="experiments/results/stratified/1t2cols/2_5g/")

    # q1.build_model5_2m("ss_gb2_5_2")
    # q1.mdl_name="ss_gb2_5_2"
    # q1.query_workload("ss_gb2_5_2",result2file="experiments/results/stratified/1t2cols/5g/")

    # q1.build_model_10m("ss_gb2_10")
    # q1.mdl_name="ss_gb2_10"
    # q1.query_workload("ss_gb2_10",result2file="experiments/results/stratified/1t2cols/10g/")

    # q1.build_model_20m("ss_gb2_20")
    # q1.mdl_name="ss_gb2_20"
    # q1.query_workload("ss_gb2_20",result2file="experiments/results/stratified/1t2cols/20g/")

    # q1.build_model_30m("ss_gb2_30")
    # q1.mdl_name="ss_gb2_30"
    # q1.query_workload("ss_gb2_30",result2file="experiments/results/stratified/1t2cols/30g/")


    q1.build_model_10m("ss_1t_10m_embedding_epoch30_node_3030_gaussian_1530_embedding_30_10m_embedding30_epoch_50_10m")
    q1.mdl_name="ss_1t_10m_embedding_epoch30_node_3030_gaussian_1530_embedding_30_10m_embedding30_epoch_50_10m"
    q1.query_workload("ss_1t_10m_embedding_epoch30_node_3030_gaussian_1530_embedding_30_10m_embedding30_epoch_50_10m",result2file="experiments/results/stratified/1t2cols/10g/")

    