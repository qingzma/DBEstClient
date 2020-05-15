# Created by Qingzhi Ma at 21/11/2019
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk

from dbestclient.executor.executor import SqlExecutor


def run():
    #
    sqlExecutor = SqlExecutor(config)
    sqlExecutor.set_table_headers("ss_sold_date_sk,ss_sold_time_sk,ss_item_sk,ss_customer_sk,ss_cdemo_sk,ss_hdemo_sk," +
                                  "ss_addr_sk,ss_store_sk,ss_promo_sk,ss_ticket_number,ss_quantity,ss_wholesale_cost," +
                                  "ss_list_price,ss_sales_price,ss_ext_discount_amt,ss_ext_sales_price," +
                                  "ss_ext_wholesale_cost,ss_ext_list_price,ss_ext_tax,ss_coupon_amt,ss_net_paid," +
                                  "ss_net_paid_inc_tax,ss_net_profit,none")
    # build_models(sqlExecutor)
    run_10k(sqlExecutor)
    run_100k(sqlExecutor)
    # sqlExecutor.execute(
    #     "create table tpcds40g_storesales_10k_ss_quantity_ss_sales_price(ss_quantity real, ss_sales_price real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000")
    # sqlExecutor.execute("select sum(ss_quantity) from tpcds40g_storesales_10k_ss_quantity_ss_sales_price_ where ss_sales_price between 50.00   and 100.00")
    # sqlExecutor.execute("select sum(ss_quantity) from tpcds40g_storesales_10k_ss_quantity_ss_sales_price_ where ss_sales_price between 100.00  and 150.00")
    # sqlExecutor.execute("select sum(ss_quantity) from tpcds40g_storesales_10k_ss_quantity_ss_sales_price_ where ss_sales_price between 150.00  and 200.00")


def build_models(sqlExecutor):
    # 10k
    sqlExecutor.execute(
        "create table tpcds40g_storesales_10k_ss_quantity_ss_sales_price(ss_quantity real, ss_sales_price real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_10k_ss_quantity_ss_net_profit(ss_quantity real, ss_net_profit real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_10k_ss_list_price_ss_quantity(ss_list_price real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_10k_ss_list_price_ss_list_price(ss_list_price real, ss_list_price real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_10k_ss_list_price_ss_coupon_amt(ss_list_price real, ss_coupon_amt real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_10k_ss_list_price_ss_wholesale_cost(ss_list_price real, ss_wholesale_cost real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_10k_ss_ext_discount_amt_ss_quantity(ss_ext_discount_amt real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_10k_ss_ext_sales_price_ss_quantity(ss_ext_sales_price real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_10k_ss_ext_list_price_ss_quantity(ss_ext_list_price real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_10k_ss_ext_tax_ss_quantity(ss_ext_tax real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_10k_ss_net_paid_ss_quantity(ss_net_paid real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_10k_ss_net_paid_inc_tax_ss_quantity(ss_net_paid_inc_tax real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_10k_ss_net_profit_ss_quantity(ss_net_profit real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000")

    sqlExecutor.set_table_headers(
        "ws_sold_date_sk,ws_sold_time_sk,ws_ship_date_sk,ws_item_sk,ws_bill_customer_sk,ws_bill_cdemo_sk," +
        "ws_bill_hdemo_sk,ws_bill_addr_sk,ws_ship_customer_sk,ws_ship_cdemo_sk,ws_ship_hdemo_sk,ws_ship_addr_sk," +
        "ws_web_page_sk,ws_web_site_sk,ws_ship_mode_sk,ws_warehouse_sk,ws_promo_sk,ws_order_number,ws_quantity," +
        "ws_wholesale_cost,ws_list_price,ws_sales_price,ws_ext_discount_amt,ws_ext_sales_price,ws_ext_wholesale_cost," +
        "ws_ext_list_price,ws_ext_tax,ws_coupon_amt,ws_ext_ship_cost,ws_net_paid,ws_net_paid_inc_tax," +
        "ws_net_paid_inc_ship,ws_net_paid_inc_ship_tax,ws_net_profit")
    sqlExecutor.execute(
        "create table tpcds40g_websales_10k_ws_quantity_ws_sales_price(ws_quantity real, ws_sales_price real) from '/data/tpcds/40G/web_sales.dat' method uniform size 10000")

    # 100k
    sqlExecutor.execute(
        "create table tpcds40g_storesales_100k_ss_quantity_ss_sales_price(ss_quantity real, ss_sales_price real) from '/data/tpcds/40G/store_sales.dat' method uniform size 100000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_100k_ss_quantity_ss_net_profit(ss_quantity real, ss_net_profit real) from '/data/tpcds/40G/store_sales.dat' method uniform size 100000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_100k_ss_list_price_ss_quantity(ss_list_price real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 100000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_100k_ss_list_price_ss_list_price(ss_list_price real, ss_list_price real) from '/data/tpcds/40G/store_sales.dat' method uniform size 100000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_100k_ss_list_price_ss_coupon_amt(ss_list_price real, ss_coupon_amt real) from '/data/tpcds/40G/store_sales.dat' method uniform size 100000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_100k_ss_list_price_ss_wholesale_cost(ss_list_price real, ss_wholesale_cost real) from '/data/tpcds/40G/store_sales.dat' method uniform size 100000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_100k_ss_ext_discount_amt_ss_quantity(ss_ext_discount_amt real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 100000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_100k_ss_ext_sales_price_ss_quantity(ss_ext_sales_price real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 100000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_100k_ss_ext_list_price_ss_quantity(ss_ext_list_price real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 100000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_100k_ss_ext_tax_ss_quantity(ss_ext_tax real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 100000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_100k_ss_net_paid_ss_quantity(ss_net_paid real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 100000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_100k_ss_net_paid_inc_tax_ss_quantity(ss_net_paid_inc_tax real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 100000")
    sqlExecutor.execute(
        "create table tpcds40g_storesales_100k_ss_net_profit_ss_quantity(ss_net_profit real, ss_quantity real) from '/data/tpcds/40G/store_sales.dat' method uniform size 100000")

    sqlExecutor.set_table_headers(
        "ws_sold_date_sk,ws_sold_time_sk,ws_ship_date_sk,ws_item_sk,ws_bill_customer_sk,ws_bill_cdemo_sk," +
        "ws_bill_hdemo_sk,ws_bill_addr_sk,ws_ship_customer_sk,ws_ship_cdemo_sk,ws_ship_hdemo_sk,ws_ship_addr_sk," +
        "ws_web_page_sk,ws_web_site_sk,ws_ship_mode_sk,ws_warehouse_sk,ws_promo_sk,ws_order_number,ws_quantity," +
        "ws_wholesale_cost,ws_list_price,ws_sales_price,ws_ext_discount_amt,ws_ext_sales_price,ws_ext_wholesale_cost," +
        "ws_ext_list_price,ws_ext_tax,ws_coupon_amt,ws_ext_ship_cost,ws_net_paid,ws_net_paid_inc_tax," +
        "ws_net_paid_inc_ship,ws_net_paid_inc_ship_tax,ws_net_profit")
    sqlExecutor.execute(
        "create table tpcds40g_websales_100k_ws_quantity_ws_sales_price(ws_quantity real, ws_sales_price real) from '/data/tpcds/40G/web_sales.dat' method uniform size 100000")


def run_10k(sqlExecutor):
    sums = [
        "select sum(ss_quantity) from tpcds40g_storesales_10k_ss_quantity_ss_sales_price where ss_sales_price between 50.00   and 100.00",
        "select sum(ss_quantity) from tpcds40g_storesales_10k_ss_quantity_ss_sales_price where ss_sales_price between 100.00  and 150.00",
        "select sum(ss_quantity) from tpcds40g_storesales_10k_ss_quantity_ss_sales_price where ss_sales_price between 150.00  and 200.00",
        "select sum(ss_quantity) from tpcds40g_storesales_10k_ss_quantity_ss_net_profit where ss_net_profit  between 0      and 2000",
        "select sum(ss_quantity) from tpcds40g_storesales_10k_ss_quantity_ss_net_profit where ss_net_profit  between 150    and 3000",
        "select sum(ss_quantity) from tpcds40g_storesales_10k_ss_quantity_ss_net_profit where ss_net_profit  between 50     and 25000", ]

    counts = [
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 1    and 20",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 21   and 40",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 41   and 60",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 61   and 80",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 81   and 100",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 0    and 5",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 6    and 10",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 11   and 15",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 16   and 20",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 21   and 25",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 26   and 30",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_list_price where ss_list_price     between 90 and 100",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_list_price where ss_list_price     between 70 and 80",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_list_price where ss_list_price     between 80 and 90",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_list_price where ss_list_price     between 100 and 110",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_list_price where ss_list_price     between 110 and 120",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_list_price where ss_list_price     between 120 and 130",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 7000    and 8000",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 8000    and 9000",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 9000    and 10000",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 10000   and 11000",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 11000   and 12000",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 12000   and 13000",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 10     and 30",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 20     and 40",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 30     and 50",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 40     and 60",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 50     and 70",
        "select count(ss_list_price) from tpcds40g_storesales_10k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 60     and 80",
    ]

    avgs = [
        "select avg(ss_ext_discount_amt) from tpcds40g_storesales_10k_ss_ext_discount_amt_ss_quantity where ss_quantity       between 1      and 20",
        "select avg(ss_ext_discount_amt) from tpcds40g_storesales_10k_ss_ext_discount_amt_ss_quantity where ss_quantity       between 21     and 40",
        "select avg(ss_ext_discount_amt) from tpcds40g_storesales_10k_ss_ext_discount_amt_ss_quantity where ss_quantity       between 41     and 60",
        "select avg(ss_ext_discount_amt) from tpcds40g_storesales_10k_ss_ext_discount_amt_ss_quantity where ss_quantity       between 61     and 80",
        "select avg(ss_ext_discount_amt) from tpcds40g_storesales_10k_ss_ext_discount_amt_ss_quantity where ss_quantity       between 81     and 100",
        "select avg(ss_ext_sales_price)  from tpcds40g_storesales_10k_ss_ext_sales_price_ss_quantity where ss_quantity       between 1      and 20",
        "select avg(ss_ext_sales_price)  from tpcds40g_storesales_10k_ss_ext_sales_price_ss_quantity where ss_quantity       between 21     and 40",
        "select avg(ss_ext_sales_price)  from tpcds40g_storesales_10k_ss_ext_sales_price_ss_quantity where ss_quantity       between 41     and 60",
        "select avg(ss_ext_sales_price)  from tpcds40g_storesales_10k_ss_ext_sales_price_ss_quantity where ss_quantity       between 61     and 80",
        "select avg(ss_ext_sales_price)  from tpcds40g_storesales_10k_ss_ext_sales_price_ss_quantity where ss_quantity       between 81     and 100",
        "select avg(ss_ext_list_price)   from tpcds40g_storesales_10k_ss_ext_list_price_ss_quantity where ss_quantity       between 1      and 20",
        "select avg(ss_ext_list_price)   from tpcds40g_storesales_10k_ss_ext_list_price_ss_quantity where ss_quantity       between 21     and 40",
        "select avg(ss_ext_list_price)   from tpcds40g_storesales_10k_ss_ext_list_price_ss_quantity where ss_quantity       between 41     and 60",
        "select avg(ss_ext_list_price)   from tpcds40g_storesales_10k_ss_ext_list_price_ss_quantity where ss_quantity       between 61     and 80",
        "select avg(ss_ext_list_price)   from tpcds40g_storesales_10k_ss_ext_list_price_ss_quantity where ss_quantity       between 81     and 100",
        "select avg(ss_ext_tax)          from tpcds40g_storesales_10k_ss_ext_tax_ss_quantity where ss_quantity       between 1      and 20",
        "select avg(ss_ext_tax)          from tpcds40g_storesales_10k_ss_ext_tax_ss_quantity where ss_quantity       between 21     and 40",
        "select avg(ss_ext_tax)          from tpcds40g_storesales_10k_ss_ext_tax_ss_quantity where ss_quantity       between 41     and 60",
        "select avg(ss_ext_tax)          from tpcds40g_storesales_10k_ss_ext_tax_ss_quantity where ss_quantity       between 61     and 80",
        "select avg(ss_ext_tax)          from tpcds40g_storesales_10k_ss_ext_tax_ss_quantity where ss_quantity       between 81     and 100",
        "select avg(ss_net_paid)         from tpcds40g_storesales_10k_ss_net_paid_ss_quantity where ss_quantity       between 1      and 20",
        "select avg(ss_net_paid)         from tpcds40g_storesales_10k_ss_net_paid_ss_quantity where ss_quantity       between 21     and 40",
        "select avg(ss_net_paid)         from tpcds40g_storesales_10k_ss_net_paid_ss_quantity where ss_quantity       between 41     and 60",
        "select avg(ss_net_paid)         from tpcds40g_storesales_10k_ss_net_paid_ss_quantity where ss_quantity       between 61     and 80",
        "select avg(ss_net_paid)         from tpcds40g_storesales_10k_ss_net_paid_ss_quantity where ss_quantity       between 81     and 100",
        "select avg(ss_net_paid_inc_tax) from tpcds40g_storesales_10k_ss_net_paid_inc_tax_ss_quantity where ss_quantity       between 1      and 20",
        "select avg(ss_net_paid_inc_tax) from tpcds40g_storesales_10k_ss_net_paid_inc_tax_ss_quantity where ss_quantity       between 21     and 40",
        "select avg(ss_net_paid_inc_tax) from tpcds40g_storesales_10k_ss_net_paid_inc_tax_ss_quantity where ss_quantity       between 41     and 60",
        "select avg(ss_net_paid_inc_tax) from tpcds40g_storesales_10k_ss_net_paid_inc_tax_ss_quantity where ss_quantity       between 61     and 80",
        "select avg(ss_net_paid_inc_tax) from tpcds40g_storesales_10k_ss_net_paid_inc_tax_ss_quantity where ss_quantity       between 81     and 100",
        "select avg(ss_net_profit)       from tpcds40g_storesales_10k_ss_net_profit_ss_quantity where ss_quantity       between 1      and 20",
        "select avg(ss_net_profit)       from tpcds40g_storesales_10k_ss_net_profit_ss_quantity where ss_quantity       between 21     and 40",
        "select avg(ss_net_profit)       from tpcds40g_storesales_10k_ss_net_profit_ss_quantity where ss_quantity       between 41     and 60",
        "select avg(ss_net_profit)       from tpcds40g_storesales_10k_ss_net_profit_ss_quantity where ss_quantity       between 61     and 80",
        "select avg(ss_net_profit)       from tpcds40g_storesales_10k_ss_net_profit_ss_quantity where ss_quantity       between 81     and 100",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 0      and 5",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 6      and 10",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 11     and 15",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 16     and 20",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 21     and 25",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_quantity where ss_quantity       between 26     and 30",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_list_price where ss_list_price     between 90 and 100",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_list_price where ss_list_price     between 70 and 80",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_list_price where ss_list_price     between 80 and 90",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_list_price where ss_list_price     between 100 and 110",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_list_price where ss_list_price     between 110 and 120",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_list_price where ss_list_price     between 120 and 130",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 7000    and 8000",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 8000    and 9000",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 9000    and 10000",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 10000   and 11000",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 11000   and 12000",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 12000   and 13000",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 10     and 30",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 20     and 40",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 30     and 50",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 40     and 60",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 50     and 70",
        "select avg(ss_list_price)       from tpcds40g_storesales_10k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 60     and 80",
        "select avg(ws_quantity)         from tpcds40g_websales_10k_ws_quantity_ws_sales_price   where ws_sales_price    between 100.00 and 150.00",
        "select avg(ws_quantity)         from tpcds40g_websales_10k_ws_quantity_ws_sales_price   where ws_sales_price    between  50.00 and 100.00",
        "select avg(ws_quantity)         from tpcds40g_websales_10k_ws_quantity_ws_sales_price   where ws_sales_price    between 150.00 and 200.00",
    ]

    # counts
    pres = []
    ts = []
    for query in counts:
        # print(query)
        p, t = sqlExecutor.execute(query)
        pres.append(p)
        ts.append(t)
    print("counts", pres)
    print("time", ts)

    # sums
    pres = []
    ts = []
    for query in sums:
        p, t = sqlExecutor.execute(query)
        pres.append(p)
        ts.append(t)
    print("sums", pres)
    print("time", ts)

    # avgs
    pres = []
    ts = []
    for query in avgs:
        p, t = sqlExecutor.execute(query)
        pres.append(p)
        ts.append(t)
    print("avgs", pres)
    print("time", ts)


def run_100k(sqlExecutor):
    sums = [
        "select sum(ss_quantity) from tpcds40g_storesales_100k_ss_quantity_ss_sales_price where ss_sales_price between 50.00   and 100.00",
        "select sum(ss_quantity) from tpcds40g_storesales_100k_ss_quantity_ss_sales_price where ss_sales_price between 100.00  and 150.00",
        "select sum(ss_quantity) from tpcds40g_storesales_100k_ss_quantity_ss_sales_price where ss_sales_price between 150.00  and 200.00",
        "select sum(ss_quantity) from tpcds40g_storesales_100k_ss_quantity_ss_net_profit where ss_net_profit  between 0      and 2000",
        "select sum(ss_quantity) from tpcds40g_storesales_100k_ss_quantity_ss_net_profit where ss_net_profit  between 150    and 3000",
        "select sum(ss_quantity) from tpcds40g_storesales_100k_ss_quantity_ss_net_profit where ss_net_profit  between 50     and 25000", ]

    counts = [
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 1    and 20",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 21   and 40",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 41   and 60",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 61   and 80",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 81   and 100",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 0    and 5",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 6    and 10",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 11   and 15",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 16   and 20",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 21   and 25",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 26   and 30",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_list_price where ss_list_price     between 90 and 100",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_list_price where ss_list_price     between 70 and 80",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_list_price where ss_list_price     between 80 and 90",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_list_price where ss_list_price     between 100 and 110",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_list_price where ss_list_price     between 110 and 120",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_list_price where ss_list_price     between 120 and 130",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 7000    and 8000",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 8000    and 9000",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 9000    and 10000",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 10000   and 11000",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 11000   and 12000",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 12000   and 13000",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 10     and 30",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 20     and 40",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 30     and 50",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 40     and 60",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 50     and 70",
        "select count(ss_list_price) from tpcds40g_storesales_100k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 60     and 80",
    ]

    avgs = [
        "select avg(ss_ext_discount_amt) from tpcds40g_storesales_100k_ss_ext_discount_amt_ss_quantity where ss_quantity       between 1      and 20",
        "select avg(ss_ext_discount_amt) from tpcds40g_storesales_100k_ss_ext_discount_amt_ss_quantity where ss_quantity       between 21     and 40",
        "select avg(ss_ext_discount_amt) from tpcds40g_storesales_100k_ss_ext_discount_amt_ss_quantity where ss_quantity       between 41     and 60",
        "select avg(ss_ext_discount_amt) from tpcds40g_storesales_100k_ss_ext_discount_amt_ss_quantity where ss_quantity       between 61     and 80",
        "select avg(ss_ext_discount_amt) from tpcds40g_storesales_100k_ss_ext_discount_amt_ss_quantity where ss_quantity       between 81     and 100",
        "select avg(ss_ext_sales_price)  from tpcds40g_storesales_100k_ss_ext_sales_price_ss_quantity where ss_quantity       between 1      and 20",
        "select avg(ss_ext_sales_price)  from tpcds40g_storesales_100k_ss_ext_sales_price_ss_quantity where ss_quantity       between 21     and 40",
        "select avg(ss_ext_sales_price)  from tpcds40g_storesales_100k_ss_ext_sales_price_ss_quantity where ss_quantity       between 41     and 60",
        "select avg(ss_ext_sales_price)  from tpcds40g_storesales_100k_ss_ext_sales_price_ss_quantity where ss_quantity       between 61     and 80",
        "select avg(ss_ext_sales_price)  from tpcds40g_storesales_100k_ss_ext_sales_price_ss_quantity where ss_quantity       between 81     and 100",
        "select avg(ss_ext_list_price)   from tpcds40g_storesales_100k_ss_ext_list_price_ss_quantity where ss_quantity       between 1      and 20",
        "select avg(ss_ext_list_price)   from tpcds40g_storesales_100k_ss_ext_list_price_ss_quantity where ss_quantity       between 21     and 40",
        "select avg(ss_ext_list_price)   from tpcds40g_storesales_100k_ss_ext_list_price_ss_quantity where ss_quantity       between 41     and 60",
        "select avg(ss_ext_list_price)   from tpcds40g_storesales_100k_ss_ext_list_price_ss_quantity where ss_quantity       between 61     and 80",
        "select avg(ss_ext_list_price)   from tpcds40g_storesales_100k_ss_ext_list_price_ss_quantity where ss_quantity       between 81     and 100",
        "select avg(ss_ext_tax)          from tpcds40g_storesales_100k_ss_ext_tax_ss_quantity where ss_quantity       between 1      and 20",
        "select avg(ss_ext_tax)          from tpcds40g_storesales_100k_ss_ext_tax_ss_quantity where ss_quantity       between 21     and 40",
        "select avg(ss_ext_tax)          from tpcds40g_storesales_100k_ss_ext_tax_ss_quantity where ss_quantity       between 41     and 60",
        "select avg(ss_ext_tax)          from tpcds40g_storesales_100k_ss_ext_tax_ss_quantity where ss_quantity       between 61     and 80",
        "select avg(ss_ext_tax)          from tpcds40g_storesales_100k_ss_ext_tax_ss_quantity where ss_quantity       between 81     and 100",
        "select avg(ss_net_paid)         from tpcds40g_storesales_100k_ss_net_paid_ss_quantity where ss_quantity       between 1      and 20",
        "select avg(ss_net_paid)         from tpcds40g_storesales_100k_ss_net_paid_ss_quantity where ss_quantity       between 21     and 40",
        "select avg(ss_net_paid)         from tpcds40g_storesales_100k_ss_net_paid_ss_quantity where ss_quantity       between 41     and 60",
        "select avg(ss_net_paid)         from tpcds40g_storesales_100k_ss_net_paid_ss_quantity where ss_quantity       between 61     and 80",
        "select avg(ss_net_paid)         from tpcds40g_storesales_100k_ss_net_paid_ss_quantity where ss_quantity       between 81     and 100",
        "select avg(ss_net_paid_inc_tax) from tpcds40g_storesales_100k_ss_net_paid_inc_tax_ss_quantity where ss_quantity       between 1      and 20",
        "select avg(ss_net_paid_inc_tax) from tpcds40g_storesales_100k_ss_net_paid_inc_tax_ss_quantity where ss_quantity       between 21     and 40",
        "select avg(ss_net_paid_inc_tax) from tpcds40g_storesales_100k_ss_net_paid_inc_tax_ss_quantity where ss_quantity       between 41     and 60",
        "select avg(ss_net_paid_inc_tax) from tpcds40g_storesales_100k_ss_net_paid_inc_tax_ss_quantity where ss_quantity       between 61     and 80",
        "select avg(ss_net_paid_inc_tax) from tpcds40g_storesales_100k_ss_net_paid_inc_tax_ss_quantity where ss_quantity       between 81     and 100",
        "select avg(ss_net_profit)       from tpcds40g_storesales_100k_ss_net_profit_ss_quantity where ss_quantity       between 1      and 20",
        "select avg(ss_net_profit)       from tpcds40g_storesales_100k_ss_net_profit_ss_quantity where ss_quantity       between 21     and 40",
        "select avg(ss_net_profit)       from tpcds40g_storesales_100k_ss_net_profit_ss_quantity where ss_quantity       between 41     and 60",
        "select avg(ss_net_profit)       from tpcds40g_storesales_100k_ss_net_profit_ss_quantity where ss_quantity       between 61     and 80",
        "select avg(ss_net_profit)       from tpcds40g_storesales_100k_ss_net_profit_ss_quantity where ss_quantity       between 81     and 100",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 0      and 5",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 6      and 10",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 11     and 15",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 16     and 20",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 21     and 25",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_quantity where ss_quantity       between 26     and 30",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_list_price where ss_list_price     between 90 and 100",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_list_price where ss_list_price     between 70 and 80",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_list_price where ss_list_price     between 80 and 90",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_list_price where ss_list_price     between 100 and 110",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_list_price where ss_list_price     between 110 and 120",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_list_price where ss_list_price     between 120 and 130",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 7000    and 8000",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 8000    and 9000",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 9000    and 10000",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 10000   and 11000",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 11000   and 12000",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_coupon_amt where ss_coupon_amt     between 12000   and 13000",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 10     and 30",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 20     and 40",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 30     and 50",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 40     and 60",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 50     and 70",
        "select avg(ss_list_price)       from tpcds40g_storesales_100k_ss_list_price_ss_wholesale_cost where ss_wholesale_cost between 60     and 80",
        "select avg(ws_quantity)         from tpcds40g_websales_100k_ws_quantity_ws_sales_price   where ws_sales_price    between 100.00 and 150.00",
        "select avg(ws_quantity)         from tpcds40g_websales_100k_ws_quantity_ws_sales_price   where ws_sales_price    between  50.00 and 100.00",
        "select avg(ws_quantity)         from tpcds40g_websales_100k_ws_quantity_ws_sales_price   where ws_sales_price    between 150.00 and 200.00",
    ]

    # counts
    pres = []
    ts = []
    for query in counts:
        p, t = sqlExecutor.execute(query)
        pres.append(p)
        ts.append(t)
    print("counts", pres)
    print("time", ts)

    # sums
    pres = []
    ts = []
    for query in sums:
        p, t = sqlExecutor.execute(query)
        pres.append(p)
        ts.append(t)
    print("sums", pres)
    print("time", ts)

    # avgs
    pres = []
    ts = []
    for query in avgs:
        p, t = sqlExecutor.execute(query)
        pres.append(p)
        ts.append(t)
    print("avgs", pres)
    print("time", ts)


if __name__ == "__main__":
    run()
