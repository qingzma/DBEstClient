create table tpcds40g_storesales_10k_ss_quantity_ss_sales_price(ss_quantity real, ss_sales_price real) from '/data/tpcds/40G/store_sales.dat' method uniform size 10000
select sum(ss_quantity) from tpcds40g_storesales_10k_ss_quantity_ss_sales_price where ss_sales_price between 50.00   and 100.00
select sum(ss_quantity) from tpcds40g_storesales_10k_ss_quantity_ss_sales_price where ss_sales_price between 100.00  and 150.00
select sum(ss_quantity) from tpcds40g_storesales_10k_ss_quantity_ss_sales_price where ss_sales_price between 150.00  and 200.00