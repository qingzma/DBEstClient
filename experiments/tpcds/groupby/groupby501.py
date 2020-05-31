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

    sqlExecutor = SqlExecutor()
    # build_501_groups(sqlExecutor)
    # build_501_groups_grid_search(sqlExecutor)
    # run_501_gogs(sqlExecutor)
    build_501_groups2(sqlExecutor)


def build_models(sqlExecutor):

    # sqlExecutor.execute(
    #     "create table ss1t_gg4_no_data(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=True, n_per_gg=127)
    # sqlExecutor.execute(
    #     "create table ss1t_gg8(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=True, n_per_gg=64)
    # sqlExecutor.execute(
    #     "create table ss1t_gg16(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=True, n_per_gg=32)
    # sqlExecutor.execute(
    #     "create table ss1t_gg1_gpu(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='gpu', b_use_gg=True, n_per_gg=512)

    # sqlExecutor.execute(
    #     "create table ss1t_gg1_cpu(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=True, n_per_gg=512)
    sqlExecutor.execute(
        "create table ss1t_gg1_cpu(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=True, n_per_gg=512)
    sqlExecutor.execute(
        "create table ss1t_gg2_cpu(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=True, n_per_gg=255)
    sqlExecutor.execute(
        "create table ss1t_gg4_cpu(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='cpu', b_use_gg=True, n_per_gg=127)
    sqlExecutor.execute(
        "create table ss1t_gg4_gpu(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size 5000000", n_mdn_layer_node=8, b_one_hot_encoding=True, b_grid_search=True, device='gpu', b_use_gg=True, n_per_gg=127)
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
    sqlExecutor.execute("set n_jobs=" + str(n_jobs)+'"')
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/sum1.txt'")
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name +
                        "  where   2451119 <=ss_sold_date_sk<= 2451483 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/sum2.txt'")
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name +
                        "  where  2451300 <=ss_sold_date_sk<= 2451665 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/sum3.txt'")
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name +
                        "  where  2451392 <=ss_sold_date_sk<= 2451757 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/sum4.txt'")
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name +
                        "  where  2451484 <=ss_sold_date_sk<= 2451849 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/sum5.txt'")
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name +
                        "  where  2451545 <=ss_sold_date_sk<= 2451910 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/sum6.txt'")
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name +
                        "  where  2451636 <=ss_sold_date_sk<= 2452000 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/sum7.txt'")
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name +
                        "  where  2451727 <=ss_sold_date_sk<= 2452091 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/sum8.txt'")
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name +
                        "  where  2451850 <=ss_sold_date_sk<= 2452214 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/sum9.txt'")
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name +
                        "  where  2451911 <=ss_sold_date_sk<= 2452275 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/sum10.txt'")
    sqlExecutor.execute("select sum(ss_sales_price)   from " + model_name +
                        "  where  2452031 <=ss_sold_date_sk<= 2452395 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/count1.txt'")
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name +
                        "  where  2451119 <=ss_sold_date_sk<= 2451483 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/count2.txt'")
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name +
                        "  where  2451300 <=ss_sold_date_sk<= 2451665 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/count3.txt'")
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name +
                        "  where  2451392 <=ss_sold_date_sk<= 2451757 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/count4.txt'")
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name +
                        "  where  2451484 <=ss_sold_date_sk<= 2451849 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/count5.txt'")
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name +
                        "  where  2451545 <=ss_sold_date_sk<= 2451910 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/count6.txt'")
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name +
                        "  where  2451636 <=ss_sold_date_sk<= 2452000 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/count7.txt'")
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name +
                        "  where  2451727 <=ss_sold_date_sk<= 2452091 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/count8.txt'")
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name +
                        "  where  2451850 <=ss_sold_date_sk<= 2452214 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/count9.txt'")
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name +
                        "  where  2451911 <=ss_sold_date_sk<= 2452275 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/count10.txt'")
    sqlExecutor.execute("select count(ss_sales_price) from " + model_name +
                        "  where  2452031 <=ss_sold_date_sk<= 2452395 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/avg1.txt'")
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name +
                        "  where  2451119 <=ss_sold_date_sk<= 2451483 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/avg2.txt'")
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name +
                        "  where  2451300 <=ss_sold_date_sk<= 2451665 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/avg3.txt'")
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name +
                        "  where  2451392 <=ss_sold_date_sk<= 2451757 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/avg4.txt'")
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name +
                        "  where  2451484 <=ss_sold_date_sk<= 2451849 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/avg5.txt'")
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name +
                        "  where  2451545 <=ss_sold_date_sk<= 2451910 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/avg6.txt'")
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name +
                        "  where  2451636 <=ss_sold_date_sk<= 2452000 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/avg7.txt'")
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name +
                        "  where  2451727 <=ss_sold_date_sk<= 2452091 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/avg8.txt'")
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name +
                        "  where  2451850 <=ss_sold_date_sk<= 2452214 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/avg9.txt'")
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name +
                        "  where  2451911 <=ss_sold_date_sk<= 2452275 group by   ss_store_sk",)
    sqlExecutor.execute(
        "set result2file='/home/u1796377/Documents/workspace/DBEstClient/experiments/results/mdn501/avg10.txt'")
    sqlExecutor.execute("select avg(ss_sales_price)   from " + model_name +
                        "  where  2452031 <=ss_sold_date_sk<= 2452395 group by   ss_store_sk",)


def build_501_groups(sqlExecutor):
    sqlExecutor.execute("set v='True'")
    sqlExecutor.execute("set device='gpu'")
    sqlExecutor.execute("set encoder='binary'")
    sqlExecutor.execute("set b_grid_search='false'")
    sqlExecutor.execute("set b_print_to_screen='false'")
    sqlExecutor.execute("set csv_split_char='|'")
    sqlExecutor.execute("set batch_size=1000")
    # sqlExecutor.execute("set table_header=" +
    #                     "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
    #                     "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
    #                     "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
    #                     "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
    #                     "ss_net_paid_inc_tax|ss_net_profit|none'"
    #                     )
    sqlExecutor.execute("set table_header=" +
                        "'ss_sold_date_sk|ss_store_sk|ss_sales_price'")

    sqlExecutor.execute("set n_mdn_layer_node=20")
    sqlExecutor.execute("set n_jobs=1")
    sqlExecutor.execute("set n_hidden_layer=1")
    sqlExecutor.execute("set n_epoch=20")
    sqlExecutor.execute("set n_gaussians_reg=3")
    sqlExecutor.execute("set n_gaussians_density=20")
    # sqlExecutor.execute("set result2file='/home/u1796377/Desktop/hah.txt'")
    sqlExecutor.execute(
        "create table ss1t_10gpu(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_10m_reduced.csv' GROUP BY ss_store_sk method uniform size  'num_points/num_of_points501.csv' ")  # num_of_points57.csv
    sqlExecutor.execute(
        "select avg(ss_sales_price)  from ss1t_10gpu where   2451119  <=ss_sold_date_sk<= 2451483    group by ss_store_sk")

    query_workload(sqlExecutor, "ss1t_10gpu", 1)


def build_501_groups2(sqlExecutor):
    sqlExecutor.execute("set v='True'")
    sqlExecutor.execute("set device='cpu'")
    sqlExecutor.execute("set encoder='binary'")
    sqlExecutor.execute("set b_grid_search='false'")
    sqlExecutor.execute("set b_print_to_screen='false'")
    sqlExecutor.execute("set csv_split_char='|'")
    sqlExecutor.execute("set batch_size=1000")
    sqlExecutor.execute("set table_header=" +
                        "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
                        "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
                        "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
                        "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
                        "ss_net_paid_inc_tax|ss_net_profit|none'"
                        )
    # sqlExecutor.execute("set table_header=" +
    #                     "'ss_sold_date_sk|ss_store_sk|ss_sales_price'")

    sqlExecutor.execute("set n_mdn_layer_node=20")
    sqlExecutor.execute("set n_jobs=2")
    sqlExecutor.execute("set n_hidden_layer=1")
    sqlExecutor.execute("set n_epoch=1")
    sqlExecutor.execute("set n_gaussians_reg=3")
    sqlExecutor.execute("set n_gaussians_density=20")
    # sqlExecutor.execute("set result2file='/home/u1796377/Desktop/hah.txt'")
    sqlExecutor.execute(
        "create table ss1t_groups2(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk,ss_quantity method uniform size 0.1 ")  # num_of_points57.csv
    
    sqlExecutor.execute("set result2file='/home/u1796377/Desktop/hah.txt'")
    sqlExecutor.execute(
        "select avg(ss_sales_price)  from ss1t_groups2 where   2451119  <=ss_sold_date_sk<= 2451483    group by ss_store_sk,ss_quantity")

    # query_workload(sqlExecutor, "ss1t_groups2", 1)


def build_501_groups_grid_search(sqlExecutor):
    sqlExecutor.execute("set v='True'")
    sqlExecutor.execute("set device='gpu'")
    sqlExecutor.execute("set encoder='binary'")
    sqlExecutor.execute("set b_grid_search='true'")
    sqlExecutor.execute("set b_print_to_screen='false'")
    sqlExecutor.execute("set csv_split_char='|'")
    sqlExecutor.execute("set batch_size=1000")
    sqlExecutor.execute("set table_header=" +
                        "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
                        "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
                        "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
                        "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
                        "ss_net_paid_inc_tax|ss_net_profit|none'"
                        )

    sqlExecutor.execute("set n_mdn_layer_node=20")
    sqlExecutor.execute("set n_jobs=1")
    sqlExecutor.execute("set n_hidden_layer=2")
    sqlExecutor.execute("set n_epoch=20")
    # sqlExecutor.execute("set n_division=50")

    # sqlExecutor.execute("set result2file='/home/u1796377/Desktop/hah.txt'")
    sqlExecutor.execute(
        "create table ss1t_501_grid_search(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_5m.csv' GROUP BY ss_store_sk method uniform size  'num_points/num_of_points501.csv' ")  # num_of_points57.csv
    sqlExecutor.execute(
        "select avg(ss_sales_price)  from ss1t_501_grid_search where   2451119  <=ss_sold_date_sk<= 2451483    group by ss_store_sk")
    query_workload(sqlExecutor, "ss1t_501_grid_search", 1)


def run_501_gogs(sqlExecutor):
    # sqlExecutor.execute("set device='cpu'")
    sqlExecutor.execute("set b_print_to_screen='False'")
    sqlExecutor.execute("set device='cpu'")
    sqlExecutor.execute("set n_jobs=1")

    sqlExecutor.execute("set b_grid_search='false'")
    sqlExecutor.execute("set csv_split_char='|'")
    # sqlExecutor.execute("set table_header=" +
    #                     "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
    #                     "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
    #                     "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
    #                     "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
    #                     "ss_net_paid_inc_tax|ss_net_profit|none'"
    #                     )
    sqlExecutor.execute("set table_header=" +
                        "'ss_sold_date_sk|ss_store_sk|ss_sales_price'")

    sqlExecutor.execute("set n_mdn_layer_node=10")
    sqlExecutor.execute("set b_use_gg='true'")
    sqlExecutor.execute("set n_per_gg=102")
    sqlExecutor.execute("set n_hidden_layer=1")
    sqlExecutor.execute("set n_epoch=20")
    sqlExecutor.execute("set b_grid_search='false'")
    sqlExecutor.execute("set n_gaussians_reg=3")
    sqlExecutor.execute("set n_gaussians_density=15")

    # sqlExecutor.execute("set result2file='/home/u1796377/Desktop/hah.txt'")
    sqlExecutor.execute(
        "create table ss1t_gogs10m_128(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/1t/ss_10m_reduced.csv' GROUP BY ss_store_sk method uniform size  'num_points/num_of_points501.csv' ")  # num_of_points57.csv
    # sqlExecutor.execute(
    #     "select avg(ss_sales_price)  from ss1t_gogs10m_128 where   2451119  <=ss_sold_date_sk<= 2451483    group by ss_store_sk")

    query_workload(sqlExecutor, "ss1t_gogs_128", 1)


if __name__ == "__main__":
    run()
    # run_dbest1()
