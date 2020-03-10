# Created by Qingzhi Ma at 10/02/2020
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
#
#
# hive -e "select ss_store_sk, count(*) from store_sales_1t where ss_sold_date_sk between 2451119  and 2451483  group by ss_store_sk;" > ~/group501counts.csv
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
        "epsabs": 20.0,  # 10
        "epsrel": 0.4,  # 0.1
        "mesh_grid_num": 15,  # 20
        "limit": 20,  # 30
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
        "create table ss1t_1m_gg_64_node2(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_1m.csv' GROUP BY ss_store_sk method uniform size 1000000",
        n_per_gg=8,n_mdn_layer_node=2)
    sqlExecutor.execute(
        "create table ss1t_1m_gg_64_node4(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_1m.csv' GROUP BY ss_store_sk method uniform size 1000000",
        n_per_gg=8, n_mdn_layer_node=4)
    sqlExecutor.execute(
        "create table ss1t_1m_gg_64_node6(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_1m.csv' GROUP BY ss_store_sk method uniform size 1000000",
        n_per_gg=8, n_mdn_layer_node=6)
    sqlExecutor.execute(
        "create table ss1t_1m_gg_64_node8(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_1m.csv' GROUP BY ss_store_sk method uniform size 1000000",
        n_per_gg=8, n_mdn_layer_node=8)
    sqlExecutor.execute(
        "create table ss1t_1m_gg_64_node10(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_1m.csv' GROUP BY ss_store_sk method uniform size 1000000",
        n_per_gg=8, n_mdn_layer_node=10)
    sqlExecutor.execute(
        "create table ss1t_1m_gg_64_node12(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_1m.csv' GROUP BY ss_store_sk method uniform size 1000000",
        n_per_gg=8, n_mdn_layer_node=12)
    sqlExecutor.execute(
        "create table ss1t_1m_gg_64_node14(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_1m.csv' GROUP BY ss_store_sk method uniform size 1000000",
        n_per_gg=8, n_mdn_layer_node=14)
    sqlExecutor.execute(
        "create table ss1t_1m_gg_64_node16(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_1m.csv' GROUP BY ss_store_sk method uniform size 1000000",
        n_per_gg=8, n_mdn_layer_node=16)
    sqlExecutor.execute(
        "create table ss1t_1m_gg_64_node18(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_1m.csv' GROUP BY ss_store_sk method uniform size 1000000",
        n_per_gg=8, n_mdn_layer_node=18)
    sqlExecutor.execute(
        "create table ss1t_1m_gg_64_node20(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_1m.csv' GROUP BY ss_store_sk method uniform size 1000000",
        n_per_gg=8, n_mdn_layer_node=20)

    # sqlExecutor.execute(
    #     "create table ss1t_1m_gg_32(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_1m.csv' GROUP BY ss_store_sk method uniform size 1000000",
    #     n_per_gg=16)
    # sqlExecutor.execute(
    #     "create table ss1t_1m_gg_16(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_1m.csv' GROUP BY ss_store_sk method uniform size 1000000",
    #     n_per_gg=32)
    # sqlExecutor.execute(
    #     "create table ss1t_1m_gg_8(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_1m.csv' GROUP BY ss_store_sk method uniform size 1000000",
    #     n_per_gg=64)
    # sqlExecutor.execute(
    #     "create table ss1t_1m_gg_4(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_1m.csv' GROUP BY ss_store_sk method uniform size 1000000",
    #     n_per_gg=127)
    # sqlExecutor.execute(
    #     "create table ss1t_1m_gg_2(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_1m.csv' GROUP BY ss_store_sk method uniform size 1000000",
    #     n_per_gg=254)


def query(sqlExecutor):
    # sqlExecutor.execute(
        # "select count(ss_sales_price)  from ss1t_1m_gg_64_node2 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
        # result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg64.txt")
    # sqlExecutor.execute(
    #     "select count(ss_sales_price)  from ss1t_1m_gg_32 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
    #     result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg32.txt")
    # sqlExecutor.execute(
    #     "select count(ss_sales_price)  from ss1t_1m_gg_16 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
    #     result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg16.txt")
    # sqlExecutor.execute(
    #     "select count(ss_sales_price)  from ss1t_1m_gg_8 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
    #     result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg8.txt")
    # sqlExecutor.execute(
    #     "select count(ss_sales_price)  from ss1t_1m_gg_4 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
    #     result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg4.txt")
    # sqlExecutor.execute(
    #     "select count(ss_sales_price)  from ss1t_1m_gg_2 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
    #     result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg2.txt")

    sqlExecutor.execute(
        "select count(ss_sales_price)  from ss1t_1m_gg_64_node2 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg64_node2.txt")
    sqlExecutor.execute(
        "select count(ss_sales_price)  from ss1t_1m_gg_64_node4 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg64_node4.txt")
    sqlExecutor.execute(
        "select count(ss_sales_price)  from ss1t_1m_gg_64_node6 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg64_node6.txt")
    sqlExecutor.execute(
        "select count(ss_sales_price)  from ss1t_1m_gg_64_node8 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg64_node8.txt")
    sqlExecutor.execute(
        "select count(ss_sales_price)  from ss1t_1m_gg_64_node10 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg64_node10.txt")
    sqlExecutor.execute(
        "select count(ss_sales_price)  from ss1t_1m_gg_64_node12 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg64_node12.txt")
    sqlExecutor.execute(
        "select count(ss_sales_price)  from ss1t_1m_gg_64_node14 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg64_node14.txt")
    sqlExecutor.execute(
        "select count(ss_sales_price)  from ss1t_1m_gg_64_node16 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg64_node16.txt")
    sqlExecutor.execute(
        "select count(ss_sales_price)  from ss1t_1m_gg_64_node18 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg64_node18.txt")
    sqlExecutor.execute(
        "select count(ss_sales_price)  from ss1t_1m_gg_64_node20 where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_gg64_node20.txt")


if __name__ == "__main__":
    run()
