# Created by Qingzhi Ma at 21/11/2019
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk

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
        "num_epoch": 200,
        "reg_type": "mdn",
        "num_gaussians": 4,
    }
    sqlExecutor = SqlExecutor(config)
    sqlExecutor.set_table_headers("ss_sold_date_sk,ss_sold_time_sk,ss_item_sk,ss_customer_sk,ss_cdemo_sk,ss_hdemo_sk," +
                                  "ss_addr_sk,ss_store_sk,ss_promo_sk,ss_ticket_number,ss_quantity,ss_wholesale_cost," +
                                  "ss_list_price,ss_sales_price,ss_ext_discount_amt,ss_ext_sales_price," +
                                  "ss_ext_wholesale_cost,ss_ext_list_price,ss_ext_tax,ss_coupon_amt,ss_net_paid," +
                                  "ss_net_paid_inc_tax,ss_net_profit,none")
    # sqlExecutor.execute(
    #     "create table tpcds40g_storesales_10k_ss_quantity_ss_sales_price(ss_quantity real, ss_sales_price real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000")
    sqlExecutor.execute("select sum(ss_quantity) from tpcds40g_storesales_10k_ss_quantity_ss_sales_price_ where ss_sales_price between 50.00   and 100.00")
    sqlExecutor.execute("select sum(ss_quantity) from tpcds40g_storesales_10k_ss_quantity_ss_sales_price_ where ss_sales_price between 100.00  and 150.00")
    sqlExecutor.execute("select sum(ss_quantity) from tpcds40g_storesales_10k_ss_quantity_ss_sales_price_ where ss_sales_price between 150.00  and 200.00")


if __name__ == "__main__":
    run()
