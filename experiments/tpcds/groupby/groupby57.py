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
    config = {
        'warehousedir': '/home/u1796377/Programs/dbestwarehouse',
        'verbose': 'True',
        'b_show_latency': 'True',
        'backend_server': 'None',
        'csv_split_char': '|',
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
    sqlExecutor = SqlExecutor(config)
    sqlExecutor.set_table_headers("ss_sold_date_sk,ss_sold_time_sk,ss_item_sk,ss_customer_sk,ss_cdemo_sk,ss_hdemo_sk," +
                                  "ss_addr_sk,ss_store_sk,ss_promo_sk,ss_ticket_number,ss_quantity,ss_wholesale_cost," +
                                  "ss_list_price,ss_sales_price,ss_ext_discount_amt,ss_ext_sales_price," +
                                  "ss_ext_wholesale_cost,ss_ext_list_price,ss_ext_tax,ss_coupon_amt,ss_net_paid," +
                                  "ss_net_paid_inc_tax,ss_net_profit,none")
    
    build_models(sqlExecutor)
    query(sqlExecutor)


def build_models(sqlExecutor):
    # 10k
    sqlExecutor.execute(
        "create table ss40g_600k_tes_gg(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk method uniform size 600000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=False, device='gpu', b_use_gg=True, n_per_gg=20)
    # "create table ss40g_600k(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk method uniform size 600000")
    # "create table ss_600k(ss_quantity real, ss_sales_price real) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk method uniform size 600000")


def query(sqlExecutor):
    sqlExecutor.execute(
        "select count(ss_sales_price)  from ss40g_600k_tes_gg where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk", n_jobs=1)
    sqlExecutor.execute(
        "select sum(ss_sales_price)  from ss40g_600k_tes_gg where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk", n_jobs=1)
    sqlExecutor.execute(
        "select avg(ss_sales_price)  from ss40g_600k_tes_gg where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk", n_jobs=1)

    # ("select count(ss_quantity)  from ss_600k where ss_sales_price between 1  and 20   group by ss_store_sk")


if __name__ == "__main__":
    run()
