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
    query_workload(sqlExecutor, 'ss1t_gg4', 8)
    # query_workload(sqlExecutor, "ss1t_gg32_cpu", 1)
    # query_workload(sqlExecutor, "ss1t_gg32_cpu", 2)
    # query_workload(sqlExecutor, "ss1t_gg32_cpu", 4)
    # query_workload(sqlExecutor, "ss1t_gg32_cpu", 8)
    # query_workload(sqlExecutor, "ss1t_gg32_cpu", 16)
    # query_workload(sqlExecutor, "ss1t_gg32_gpu", 1)
    # query_workload(sqlExecutor, "ss1t_gg32_gpu", 2)
    # query_workload(sqlExecutor, "ss1t_gg32_gpu", 4)
    # query_workload(sqlExecutor, "ss1t_gg32_gpu", 8)
    # query_workload(sqlExecutor, "ss1t_gg32_gpu", 16)

    # query_workload(sqlExecutor, "ss1t_gg64_cpu", 1)
    # query_workload(sqlExecutor, "ss1t_gg64_cpu", 2)
    # query_workload(sqlExecutor, "ss1t_gg64_cpu", 4)
    # query_workload(sqlExecutor, "ss1t_gg64_cpu", 8)
    # query_workload(sqlExecutor, "ss1t_gg64_cpu", 16)
    # query_workload(sqlExecutor, "ss1t_gg64_gpu", 1)
    # query_workload(sqlExecutor, "ss1t_gg64_gpu", 2)
    # query_workload(sqlExecutor, "ss1t_gg64_gpu", 4)
    # query_workload(sqlExecutor, "ss1t_gg64_gpu", 8)
    # query_workload(sqlExecutor, "ss1t_gg64_gpu", 16)


def build_models(sqlExecutor):

    # sqlExecutor.execute(
    #     "create table ss1t_gg4_no_data(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=True, n_per_gg=127)
    # sqlExecutor.execute(
    #     "create table ss1t_gg8(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=True, n_per_gg=64)
    # sqlExecutor.execute(
    #     "create table ss1t_gg16(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=True, n_per_gg=32)
    # sqlExecutor.execute(
    #     "create table ss1t_gg1_gpu(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='gpu', b_use_gg=True, n_per_gg=512)
    sqlExecutor.execute(
        "create table ss1t_gg2_gpu(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='gpu', b_use_gg=True, n_per_gg=256)
    # sqlExecutor.execute(
    #     "create table ss1t_gg1_cpu(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=True, n_per_gg=512)
    sqlExecutor.execute(
        "create table ss1t_gg2_cpu(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=True, n_per_gg=256)
    # sqlExecutor.execute(
    #     "create table ss1t_gg2(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=True, n_per_gg=254)
    # sqlExecutor.execute(
    #     "create table ss1t_no_gg(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=False, n_per_gg=254)
    sqlExecutor.execute(
        "create table ss1t_gg32_cpu(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=True, n_per_gg=16)
    sqlExecutor.execute(
        "create table ss1t_gg64_cpu(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=True, n_per_gg=8)
    sqlExecutor.execute(
        "create table ss1t_gg32_gpu(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='gpu', b_use_gg=True, n_per_gg=16)
    sqlExecutor.execute(
        "create table ss1t_gg64_gpu(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='gpu', b_use_gg=True, n_per_gg=8)


def query(sqlExecutor):
    sqlExecutor.execute(
        'select sum(ss_sales_price)   from ss1t_gg32_cpu where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk', n_jobs=1)
    sqlExecutor.execute(
        'select sum(ss_sales_price)   from ss1t_gg32_cpu where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk', n_jobs=2)
    sqlExecutor.execute(
        'select sum(ss_sales_price)   from ss1t_gg32_cpu where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk', n_jobs=4)
    sqlExecutor.execute(
        'select sum(ss_sales_price)   from ss1t_gg32_cpu where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk', n_jobs=8)
    sqlExecutor.execute(
        'select sum(ss_sales_price)   from ss1t_gg32_cpu where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk', n_jobs=16)

    sqlExecutor.execute(
        'select sum(ss_sales_price)   from ss1t_gg32_gpu where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk', n_jobs=1)
    sqlExecutor.execute(
        'select sum(ss_sales_price)   from ss1t_gg32_gpu where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk', n_jobs=2)
    sqlExecutor.execute(
        'select sum(ss_sales_price)   from ss1t_gg32_gpu where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk', n_jobs=4)
    sqlExecutor.execute(
        'select sum(ss_sales_price)   from ss1t_gg32_gpu where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk', n_jobs=8)
    sqlExecutor.execute(
        'select sum(ss_sales_price)   from ss1t_gg32_gpu where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk', n_jobs=16)

    # sqlExecutor.execute('select sum(ss_sales_price)   from ss1t_gg32_cpu where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk',
    #                     result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/ss1t_gg32_cpu.txt", n_jobs=1)


def query_workload(sqlExecutor, model_name, n_jobs):

    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/sum1_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451300 and 2451665 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/sum2_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451392 and 2451757 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/sum3_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451484 and 2451849 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/sum4_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451545 and 2451910 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/sum5_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451636 and 2452000 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/sum6_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451727 and 2452091 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/sum7_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451850 and 2452214 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/sum8_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451911 and 2452275 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/sum9_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2452031 and 2452395 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/sum10_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name + "  where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count1_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name + "  where ss_sold_date_sk between 2451300 and 2451665 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count2_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name + "  where ss_sold_date_sk between 2451392 and 2451757 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count3_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name + "  where ss_sold_date_sk between 2451484 and 2451849 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count4_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name + "  where ss_sold_date_sk between 2451545 and 2451910 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count5_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name + "  where ss_sold_date_sk between 2451636 and 2452000 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count6_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name + "  where ss_sold_date_sk between 2451727 and 2452091 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count7_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name + "  where ss_sold_date_sk between 2451850 and 2452214 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count8_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name + "  where ss_sold_date_sk between 2451911 and 2452275 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count9_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name + "  where ss_sold_date_sk between 2452031 and 2452395 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/count10_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451119 and 2451483 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/avg1_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451300 and 2451665 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/avg2_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451392 and 2451757 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/avg3_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451484 and 2451849 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/avg4_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451545 and 2451910 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/avg5_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451636 and 2452000 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/avg6_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451727 and 2452091 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/avg7_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451850 and 2452214 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/avg8_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2451911 and 2452275 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/avg9_" + model_name + ".txt", n_jobs=n_jobs)
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name + "  where ss_sold_date_sk between 2452031 and 2452395 group by   ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/avg10_" + model_name + ".txt", n_jobs=n_jobs)


def run_dbest1():
    config = {
        'warehousedir': '/home/u1796377/Programs/dbestwarehouse',
        'verbose': 'True',
        'b_show_latency': 'True',
        'backend_server': 'None',
        'csv_split_char': '|',
        "epsabs": 10.0,  # 20
        "epsrel": 0.1,  # 0.4
        "mesh_grid_num": 15,  # 20
        "limit": 20,  # 30
        # "b_reg_mean":'True',
        "num_epoch": 400,
        "reg_type": "qreg",
        "density_type": "kde",
        "num_gaussians": 4,
    }
    sqlExecutor = SqlExecutor(config)
    sqlExecutor.set_table_headers("ss_sold_date_sk,ss_sold_time_sk,ss_item_sk,ss_customer_sk,ss_cdemo_sk,ss_hdemo_sk," +
                                  "ss_addr_sk,ss_store_sk,ss_promo_sk,ss_ticket_number,ss_quantity,ss_wholesale_cost," +
                                  "ss_list_price,ss_sales_price,ss_ext_discount_amt,ss_ext_sales_price," +
                                  "ss_ext_wholesale_cost,ss_ext_list_price,ss_ext_tax,ss_coupon_amt,ss_net_paid," +
                                  "ss_net_paid_inc_tax,ss_net_profit,none")
    sqlExecutor.execute(
        "create table ss1t_5m_qreg(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000")
    sqlExecutor.execute("select count(ss_sales_price)  from ss1t_5m_qreg where ss_sold_date_sk between 2451119  and 2451483   group by ss_store_sk",
                        result2file="/home/u1796377/Projects/DBEstClient/experiments/results/mdn501/ss1t_5m_qreg.txt")
    # build_models(sqlExecutor)
    # query(sqlExecutor)


if __name__ == "__main__":
    run()
    # run_dbest1()
