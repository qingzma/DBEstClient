# Created by Qingzhi Ma at 27/01/2020
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
#
#
# hive -e "select ss_store_sk, count(*) from store_sales_40g where ss_sold_date_sk between 2451119  and 2451483 group by ss_store_sk;" > ~/group57counts.csv
#
#
from dbestclient.executor.executor import SqlExecutor


def run():
    sqlExecutor = SqlExecutor()
    sqlExecutor.execute("set v='True'")
    sqlExecutor.execute("set n_jobs=8")
    sqlExecutor.execute("set device='cpu'")
    sqlExecutor.execute("set encoder='binary'")
    sqlExecutor.execute("set b_grid_search='False'")
    sqlExecutor.execute("set b_print_to_screen='true'")
    sqlExecutor.execute("set csv_split_char='|'")
    sqlExecutor.execute("set batch_size=1000")
    sqlExecutor.execute("set table_header=" +
                        "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
                        "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
                        "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
                        "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
                        "ss_net_paid_inc_tax|ss_net_profit|none'"
                        )
    # run_2_groupby(sqlExecutor)
    build_models(sqlExecutor)
    query(sqlExecutor)
    # run_57_groups(sqlExecutor)
    # run_57_gogs(sqlExecutor)


def build_models(sqlExecutor):
    # 10k
    # ,ss_quantity  , ss_coupon_amt categorical
    sqlExecutor.execute(
        "create table ss40g(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk,ss_quantity method uniform size 600 ")

    # sqlExecutor.execute(
    #     "create table ss40g_no_categorical(ss_sales_price real, ss_sold_date_sk real,) from '/data/tpcds/40G/ss_1k.csv' GROUP BY ss_store_sk method uniform size 'num_of_points57.csv'")


def query(sqlExecutor):
    sqlExecutor.execute("set b_print_to_screen='false'")
    sqlExecutor.execute("set n_jobs=2")
    sqlExecutor.execute(
        "select avg(ss_sales_price)  from ss40g where   2451119  <=ss_sold_date_sk<= 2451483 and ss_coupon_amt='' and ss_quantity=''   group by ss_store_sk",)
    sqlExecutor.execute("set n_jobs=2")
    sqlExecutor.execute(
        "select avg(ss_sales_price)  from ss40g where   2451119  <=ss_sold_date_sk<= 2451483 and ss_coupon_amt='' and ss_quantity=''   group by ss_store_sk",)
    sqlExecutor.execute("set n_jobs=2")
    sqlExecutor.execute(
        "select avg(ss_sales_price)  from ss40g where   2451119  <=ss_sold_date_sk<= 2451483 and ss_coupon_amt='' and ss_quantity=''   group by ss_store_sk",)
    # sqlExecutor.execute("set n_jobs=4")
    # sqlExecutor.execute(
    #     "select avg(ss_sales_price)  from ss40g_categorical_full where   2451119  <=ss_sold_date_sk<= 2451483 and ss_coupon_amt=''  and ss_quantity=''  group by ss_store_sk",)
    # sqlExecutor.execute("set n_jobs=8")
    # sqlExecutor.execute(
    #     "select avg(ss_sales_price)  from ss40g_categorical_full where   2451119  <=ss_sold_date_sk<= 2451483 and ss_coupon_amt=''  and ss_quantity=''  group by ss_store_sk",)


def run_2_groupby(sqlExecutor):
    sqlExecutor.execute("set b_print_to_screen='false'")
    # sqlExecutor.execute("set device='cpu'")
    sqlExecutor.execute(
        "create table ss40g_gb2(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk,ss_quantity method uniform size 600000 ")  # ,ss_quantity
    sqlExecutor.execute("set n_jobs=1")
    sqlExecutor.execute(
        "select count(ss_sales_price)  from ss40g_gb2 where   2451119  <=ss_sold_date_sk<= 2451483   group by ss_store_sk,ss_quantity;")
    sqlExecutor.execute("set n_jobs=2")
    sqlExecutor.execute(
        "select count(ss_sales_price)  from ss40g_gb2 where   2451119  <=ss_sold_date_sk<= 2451483   group by ss_store_sk,ss_quantity;")
    sqlExecutor.execute("set n_jobs=4")
    sqlExecutor.execute(
        "select count(ss_sales_price)  from ss40g_gb2 where   2451119  <=ss_sold_date_sk<= 2451483   group by ss_store_sk,ss_quantity;")
    # sqlExecutor.execute("set n_jobs=8")
    # sqlExecutor.execute(
    #     "select count(ss_sales_price)  from ss40g_gb2 where   2451119  <=ss_sold_date_sk<= 2451483   group by ss_store_sk,ss_quantity;")
    # sqlExecutor.execute("set n_jobs=16")
    # sqlExecutor.execute(
    #     "select count(ss_sales_price)  from ss40g_gb2 where   2451119  <=ss_sold_date_sk<= 2451483   group by ss_store_sk,ss_quantity;")
    # sqlExecutor.execute("set n_jobs=20")
    # sqlExecutor.execute(
    #     "select count(ss_sales_price)  from ss40g_gb2 where   2451119  <=ss_sold_date_sk<= 2451483   group by ss_store_sk,ss_quantity;")


def run_57_groups(sqlExecutor):
    sqlExecutor.execute("set b_print_to_screen='False'")
    sqlExecutor.execute("set n_mdn_layer_node=10")
    sqlExecutor.execute("set n_jobs=1")
    sqlExecutor.execute("set n_hidden_layer=2")
    sqlExecutor.execute("set n_epoch=20")
    sqlExecutor.execute("set b_grid_search='true'")

    sqlExecutor.execute("set result2file='/home/u1796377/Desktop/hah.txt'")
    sqlExecutor.execute(
        "create table ss40g_57_node10_hidden2_grid_search(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk method uniform size  600 ")  # num_of_points57.csv
    sqlExecutor.execute(
        "select avg(ss_sales_price)  from ss40g_57_node10_hidden2_grid_search where   2451119  <=ss_sold_date_sk<= 2451483    group by ss_store_sk")
    # sqlExecutor.execute("set n_jobs=2")
    # sqlExecutor.execute(
    #     "select avg(ss_sales_price)  from ss40g_57 where   2451119  <=ss_sold_date_sk<= 2451483    group by ss_store_sk",)
    # sqlExecutor.execute("set n_jobs=4")
    # sqlExecutor.execute(
    #     "select avg(ss_sales_price)  from ss40g_57 where   2451119  <=ss_sold_date_sk<= 2451483    group by ss_store_sk",)
    # sqlExecutor.execute("set n_jobs=8")
    # sqlExecutor.execute(
    #     "select avg(ss_sales_price)  from ss40g_57 where   2451119  <=ss_sold_date_sk<= 2451483    group by ss_store_sk",)


def run_57_gogs(sqlExecutor):
    # sqlExecutor.execute("set device='cpu'")
    sqlExecutor.execute("set b_print_to_screen='False'")
    sqlExecutor.execute("set n_mdn_layer_node=10")
    sqlExecutor.execute("set n_jobs=1")
    sqlExecutor.execute("set n_hidden_layer=2")
    sqlExecutor.execute("set n_epoch=20")
    sqlExecutor.execute("set b_grid_search='true'")

    sqlExecutor.execute("set b_use_gg='true'")
    sqlExecutor.execute("set n_per_gg=30")
    sqlExecutor.execute("set b_grid_search='false'")
    sqlExecutor.execute("set n_gaussians_reg=3")
    sqlExecutor.execute("set n_gaussians_density=20")

    # sqlExecutor.execute("set result2file='/home/u1796377/Desktop/hah.txt'")
    sqlExecutor.execute(
        "create table ss40g_57_gog(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk method uniform size  600 ")  # num_of_points57.csv
    sqlExecutor.execute(
        "select avg(ss_sales_price)  from ss40g_57_gog where   2451119  <=ss_sold_date_sk<= 2451483    group by ss_store_sk")

    # sqlExecutor.execute("set n_jobs=2")
    # sqlExecutor.execute(
    #     "select avg(ss_sales_price)  from ss40g_57_gog where   2451119  <=ss_sold_date_sk<= 2451483    group by ss_store_sk")


if __name__ == "__main__":
    run()
