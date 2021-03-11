#
# Created by Qingzhi Ma on Wed May 13 2020
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
import unittest

from dbestclient.executor.executor import SqlExecutor


class TestTpcDs(unittest.TestCase):

    # def test_simple_model(self):
    #     sqlExecutor = SqlExecutor()
    #     sqlExecutor.execute("set n_epoch=10")
    #     sqlExecutor.execute("set reg_type='mdn'")
    #     sqlExecutor.execute("set density_type='mdn'")
    #     sqlExecutor.execute("set b_grid_search='False'")
    #     sqlExecutor.execute("set csv_split_char='|'")
    #     sqlExecutor.execute("set table_header=" +
    #                         "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
    #                         "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
    #                         "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
    #                         "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
    #                         "ss_net_paid_inc_tax|ss_net_profit|none'"
    #                         )
    #     sqlExecutor.execute(
    #         "create table test_ss40g_sm(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/40G/ss_600k.csv'  method uniform size 6000") #scale data num_of_points2.csv
    #     predictions = sqlExecutor.execute(
    #         "select avg(ss_sales_price)  from test_ss40g_sm where   2451119  <=ss_sold_date_sk<= 2451483 ")
    #     sqlExecutor.execute("drop table test_ss40g_sm")
    #     self.assertTrue(predictions)

    def test_groupbys_range_no_categorical_gb2(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        # sqlExecutor.execute("set encoder='binary'")
        sqlExecutor.execute("set n_epoch=2")
        sqlExecutor.execute(
            "set table_header="
            + "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|"
            + "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|"
            + "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|"
            + "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|"
            + "ss_net_paid_inc_tax|ss_net_profit|none'"
        )
        sqlExecutor.execute("drop table test_ss40g_groupbys_range_no_categorical_gb2")
        sqlExecutor.execute(
            "create table test_ss40g_groupbys_range_no_categorical_gb2(ss_sales_price real, ss_sold_date_sk real) from 'data/tpcds/40G/ss_100.csv' GROUP BY ss_store_sk,ss_coupon_amt method uniform size 100"
        )
        predictions = sqlExecutor.execute(
            "select avg(ss_sales_price)  from test_ss40g_groupbys_range_no_categorical_gb2 where   2451119  <=ss_sold_date_sk<= 2451483  group by ss_store_sk,ss_coupon_amt"
        )
        sqlExecutor.execute("drop table test_ss40g_groupbys_range_no_categorical_gb2")
        self.assertFalse(predictions.empty)

    def test_groupbys_range_no_categorical_gb1(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        # sqlExecutor.execute("set encoder='binary'")
        sqlExecutor.execute("set n_epoch=2")
        sqlExecutor.execute(
            "set table_header="
            + "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|"
            + "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|"
            + "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|"
            + "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|"
            + "ss_net_paid_inc_tax|ss_net_profit|none'"
        )
        sqlExecutor.execute("drop table test_ss40g_groupbys_range_no_categorical_gb1")
        sqlExecutor.execute(
            "create table test_ss40g_groupbys_range_no_categorical_gb1(ss_sales_price real, ss_sold_date_sk real) from 'data/tpcds/40G/ss_100.csv' GROUP BY ss_store_sk method uniform size 100"
        )
        predictions = sqlExecutor.execute(
            "select avg(ss_sales_price)  from test_ss40g_groupbys_range_no_categorical_gb1 where   2451119  <=ss_sold_date_sk<= 2451483  group by ss_store_sk"
        )
        sqlExecutor.execute("drop table test_ss40g_groupbys_range_no_categorical_gb1")
        self.assertFalse(predictions.empty)

    def test_groupbys_range_no_categorical_gb1_stratified(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set n_epoch=2")
        sqlExecutor.execute("set encoder='embedding'")
        sqlExecutor.execute(
            "set table_header="
            + "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|"
            + "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|"
            + "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|"
            + "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|"
            + "ss_net_paid_inc_tax|ss_net_profit|none'"
        )
        sqlExecutor.execute("drop table test_ss40g_groupbys_range_no_categorical")
        sqlExecutor.execute(
            "create table test_ss40g_groupbys_range_no_categorical(ss_sales_price real, ss_sold_date_sk real) from 'data/tpcds/40G/ss_100.csv' GROUP BY ss_store_sk method stratified size 100"
        )
        predictions = sqlExecutor.execute(
            "select avg(ss_sales_price)  from test_ss40g_groupbys_range_no_categorical where   2451119  <=ss_sold_date_sk<= 2451483  group by ss_store_sk"
        )  # ss_coupon_amt
        sqlExecutor.execute("drop table test_ss40g_groupbys_range_no_categorical")
        self.assertFalse(predictions.empty)

    def test_groupbys_range_no_categorical_gb2_stratified(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set n_epoch=2")
        sqlExecutor.execute("set encoder='embedding'")
        sqlExecutor.execute(
            "set table_header="
            + "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|"
            + "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|"
            + "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|"
            + "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|"
            + "ss_net_paid_inc_tax|ss_net_profit|none'"
        )
        sqlExecutor.execute(
            "drop table test_ss40g_groupbys_range_no_categorical_gb2_stratified"
        )
        sqlExecutor.execute(
            "create table test_ss40g_groupbys_range_no_categorical_gb2_stratified(ss_sales_price real, ss_sold_date_sk real) from 'data/tpcds/40G/ss_100.csv' GROUP BY ss_store_sk,ss_coupon_amt method stratified size 100"
        )
        predictions = sqlExecutor.execute(
            "select avg(ss_sales_price)  from test_ss40g_groupbys_range_no_categorical_gb2_stratified where   2451119  <=ss_sold_date_sk<= 2451483  group by ss_store_sk,ss_coupon_amt"
        )  # ss_coupon_amt
        sqlExecutor.execute(
            "drop table test_ss40g_groupbys_range_no_categorical_gb2_stratified"
        )
        self.assertFalse(predictions.empty)
    
    def test_groupbys_range_no_categorical_gb2_stratified_sample_only(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set n_epoch=2")
        sqlExecutor.execute("set encoder='embedding'")
        sqlExecutor.execute(
            "set table_header="
            + "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|"
            + "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|"
            + "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|"
            + "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|"
            + "ss_net_paid_inc_tax|ss_net_profit|none'"
        )
        sqlExecutor.execute(
            "drop table test_ss40g_groupbys_range_no_categorical_gb2_stratified"
        )

        sqlExecutor.execute("set sampling_only='True'")

        sqlExecutor.execute(
            "create table test_ss40g_groupbys_range_no_categorical_gb2_stratified(ss_sales_price real, ss_sold_date_sk real) from 'data/tpcds/40G/ss_100.csv' GROUP BY ss_store_sk,ss_coupon_amt method stratified size 100"
        )

        sqlExecutor.execute("set sampling_only='False'")

        # predictions = sqlExecutor.execute(
        #     "select avg(ss_sales_price)  from test_ss40g_groupbys_range_no_categorical_gb2_stratified where   2451119  <=ss_sold_date_sk<= 2451483  group by ss_store_sk,ss_coupon_amt"
        # )  # ss_coupon_amt
        sqlExecutor.execute(
            "drop table test_ss40g_groupbys_range_no_categorical_gb2_stratified"
        )
        # self.assertFalse(predictions.empty)

    # def test_categorical(self):
    #     sqlExecutor = SqlExecutor()
    #     sqlExecutor.execute("set b_grid_search='False'")
    #     sqlExecutor.execute("set csv_split_char='|'")
    # sqlExecutor.execute("set n_epoch=2")
    #     sqlExecutor.execute("set table_header=" +
    #                         "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
    #                         "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
    #                         "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
    #                         "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
    #                         "ss_net_paid_inc_tax|ss_net_profit|none'"
    #                         )
    #     sqlExecutor.execute(
    #         "create table test_ss40g_categorical(ss_sales_price real, ss_sold_date_sk real, ss_coupon_amt categorical) from 'data/tpcds/40G/ss_1k.csv' GROUP BY ss_store_sk method uniform size 1000")
    #     predictions = sqlExecutor.execute(
    #         "select avg(ss_sales_price)  from test_ss40g_categorical where   2451119  <=ss_sold_date_sk<= 2451483 and ss_coupon_amt=''  group by ss_store_sk")
    #     sqlExecutor.execute("drop table test_ss40g_categorical")
    #     self.assertTrue(predictions)

    def test_categorical_one_model(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set encoder='embedding'")
        sqlExecutor.execute("set n_epoch=2")
        sqlExecutor.execute("set one_model='true'")
        sqlExecutor.execute(
            "set table_header="
            + "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|"
            + "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|"
            + "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|"
            + "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|"
            + "ss_net_paid_inc_tax|ss_net_profit|none'"
        )
        sqlExecutor.execute("drop table test_ss40g_categorical_one_model")
        sqlExecutor.execute(
            "create table test_ss40g_categorical_one_model(ss_sales_price real, ss_sold_date_sk real,  ss_coupon_amt categorical, ) from 'data/tpcds/40G/ss_100.csv' GROUP BY ss_store_sk method uniform size 100"
        )  # ss_ext_discount_amt categorical
        predictions = sqlExecutor.execute(
            "select count(ss_sales_price)  from test_ss40g_categorical_one_model where   2451119  <=ss_sold_date_sk<= 2451483 and ss_coupon_amt='0.00'   group by ss_store_sk"
        )  # and ss_ext_discount_amt='0.00'
        sqlExecutor.execute("drop table test_ss40g_categorical_one_model")
        self.assertFalse(predictions.empty)

    def test_categorical_one_model_stratified(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set n_epoch=2")
        sqlExecutor.execute("set encoder='embedding'")
        sqlExecutor.execute("set one_model='true'")
        sqlExecutor.execute(
            "set table_header="
            + "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|"
            + "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|"
            + "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|"
            + "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|"
            + "ss_net_paid_inc_tax|ss_net_profit|none'"
        )
        sqlExecutor.execute("drop table test_ss40g_categorical_one_model_stratified")
        sqlExecutor.execute(
            "create table test_ss40g_categorical_one_model_stratified(ss_sales_price real, ss_sold_date_sk real,  ss_coupon_amt categorical, ) from 'data/tpcds/40G/ss_100.csv' GROUP BY ss_store_sk method stratified size 100"
        )  # ss_ext_discount_amt categorical
        predictions = sqlExecutor.execute(
            "select count(ss_sales_price)  from test_ss40g_categorical_one_model_stratified where   2451119  <=ss_sold_date_sk<= 2451483 and ss_coupon_amt='18.56'   group by ss_store_sk"
        )  # 18.56
        sqlExecutor.execute("drop table test_ss40g_categorical_one_model_stratified")
        self.assertFalse(predictions.empty)

    # def test_gogs_no_categorical(self):
    #     sqlExecutor = SqlExecutor()
    #     sqlExecutor.execute("set b_grid_search='False'")
    #     sqlExecutor.execute("set b_use_gg='True'")
    #     sqlExecutor.execute("set n_per_gg=10")
    #     sqlExecutor.execute("set csv_split_char='|'")
    #     sqlExecutor.execute("set table_header=" +
    #                         "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
    #                         "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
    #                         "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
    #                         "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
    #                         "ss_net_paid_inc_tax|ss_net_profit|none'"
    #                         )
    #     sqlExecutor.execute(
    #         "create table test_ss40g_gogs_no_categorical(ss_sales_price real, ss_sold_date_sk real) from 'data/tpcds/40G/ss_100.csv' GROUP BY ss_store_sk method uniform size 100")
    #     predictions = sqlExecutor.execute(
    #         "select avg(ss_sales_price)  from test_ss40g_gogs_no_categorical where   2451119  <=ss_sold_date_sk<= 2451483  group by ss_store_sk")
    #     sqlExecutor.execute("drop table test_ss40g_gogs_no_categorical")
    #     self.assertFalse(predictions.empty)

    # def test_gogs_categorical(self):
    #     sqlExecutor = SqlExecutor()
    #     sqlExecutor.execute("set b_grid_search='False'")
    #     sqlExecutor.execute("set b_use_gg='True'")
    #     sqlExecutor.execute("set n_per_gg=10")
    #     sqlExecutor.execute("set csv_split_char='|'")
    #     sqlExecutor.execute("set table_header=" +
    #                         "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
    #                         "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
    #                         "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
    #                         "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
    #                         "ss_net_paid_inc_tax|ss_net_profit|none'"
    #                         )
    #     sqlExecutor.execute("drop table test_ss40g_gogs_categorical")
    #     sqlExecutor.execute(
    #         "create table test_ss40g_gogs_categorical(ss_sales_price real, ss_sold_date_sk real, ss_coupon_amt categorical) from 'data/tpcds/40G/ss_100.csv' GROUP BY ss_store_sk method uniform size 100")
    #     predictions = sqlExecutor.execute(
    #         "select avg(ss_sales_price)  from test_ss40g_gogs_categorical where   2451119  <=ss_sold_date_sk<= 2451483 and ss_coupon_amt='' group by ss_store_sk")
    #     sqlExecutor.execute("drop table test_ss40g_gogs_categorical")
    #     self.assertFalse(predictions.empty)

    def test_drop_clause(self):
        sqlExecutor = SqlExecutor()
        status = sqlExecutor.execute("drop table model2drop")
        self.assertFalse(status)

    def test_embedding(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set n_epoch=2")
        sqlExecutor.execute("set encoder='embedding'")
        sqlExecutor.execute(
            "set table_header="
            + "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|"
            + "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|"
            + "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|"
            + "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|"
            + "ss_net_paid_inc_tax|ss_net_profit|none'"
        )
        sqlExecutor.execute(
            "create table test_ss40g_embedding(ss_sales_price real, ss_sold_date_sk real) from 'data/tpcds/10g/ss_10g_100.csv' GROUP BY ss_store_sk method uniform size 100"
        )
        predictions = sqlExecutor.execute(
            "select avg(ss_sales_price)  from test_ss40g_embedding where   2451119  <=ss_sold_date_sk<= 2451483  group by ss_store_sk"
        )
        sqlExecutor.execute("drop table test_ss40g_embedding")
        self.assertFalse(predictions.empty)

    # def test_no_continuous(self):
    #     sqlExecutor = SqlExecutor()
    #     sqlExecutor.execute("set b_grid_search='False'")
    #     sqlExecutor.execute("set csv_split_char='|'")
    # sqlExecutor.execute("set n_epoch=2")
    #     sqlExecutor.execute("set encoder='binary'")
    #     sqlExecutor.execute("set table_header=" +
    #                         "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
    #                         "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
    #                         "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
    #                         "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
    #                         "ss_net_paid_inc_tax|ss_net_profit|none'"
    #                         )
    #     sqlExecutor.execute("drop table test_ss40g_no_continuous")
    #     sqlExecutor.execute(
    #         "create table test_ss40g_no_continuous(ss_sales_price real) from 'data/tpcds/10g/ss_10g_100.csv' GROUP BY ss_store_sk method uniform size 100")  # , ss_coupon_amt real
    #     sqlExecutor.execute(
    #         "select ss_store_sk, avg(ss_sales_price)  from test_ss40g_no_continuous  group by ss_store_sk")
    #     sqlExecutor.execute("drop table test_ss40g_no_continuous1")
    #     self.assertTrue(True)

    def test_no_continuous_categorical_1(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set n_epoch=2")
        # sqlExecutor.execute("set encoder='binary'")
        sqlExecutor.execute(
            "set table_header="
            + "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|"
            + "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|"
            + "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|"
            + "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|"
            + "ss_net_paid_inc_tax|ss_net_profit|none'"
        )
        sqlExecutor.execute("drop table test_ss40g_no_continuous_1_column")
        sqlExecutor.execute(
            "create table test_ss40g_no_continuous_1_column(ss_sales_price real, ss_coupon_amt categorical) from 'data/tpcds/10g/ss_10g_100.csv' GROUP BY ss_store_sk method uniform size 100"
        )
        results = sqlExecutor.execute(
            "select ss_store_sk, avg(ss_sales_price)  from test_ss40g_no_continuous_1_column where ss_coupon_amt='143.91'  group by ss_store_sk"
        )
        sqlExecutor.execute("drop table test_ss40g_no_continuous_1_column")
        self.assertFalse(results.empty)

    def test_no_continuous_categorical_2(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set n_epoch=2")
        sqlExecutor.execute("set encoder='embedding'")
        sqlExecutor.execute(
            "set table_header="
            + "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|"
            + "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|"
            + "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|"
            + "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|"
            + "ss_net_paid_inc_tax|ss_net_profit|none'"
        )
        sqlExecutor.execute("drop table test_ss40g_no_continuous_2_columns")
        sqlExecutor.execute(
            "create table test_ss40g_no_continuous_2_columns(ss_sales_price real, ss_coupon_amt categorical,ss_customer_sk categorical) from 'data/tpcds/10g/ss_10g_100.csv' GROUP BY ss_store_sk method uniform size 100"
        )
        results = sqlExecutor.execute(
            "select ss_store_sk, avg(ss_sales_price)  from test_ss40g_no_continuous_2_columns where ss_coupon_amt='103.67'  and ss_customer_sk='415915' group by ss_store_sk"
        )
        sqlExecutor.execute("drop table test_ss40g_no_continuous_2_columns")
        self.assertFalse(results.empty)

    def test_no_continuous_categorical_one_model_uniform(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set n_epoch=2")
        sqlExecutor.execute("set encoder='embedding'")
        sqlExecutor.execute("set one_model='true'")
        sqlExecutor.execute(
            "set table_header="
            + "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|"
            + "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|"
            + "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|"
            + "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|"
            + "ss_net_paid_inc_tax|ss_net_profit|none'"
        )
        sqlExecutor.execute("drop table test_ss40g_no_continuous_1_column_1_model")
        sqlExecutor.execute(
            "create table test_ss40g_no_continuous_1_column_1_model(ss_sales_price real, ss_coupon_amt categorical) from 'data/tpcds/10g/ss_10g_100.csv' GROUP BY ss_store_sk method uniform size 100"
        )
        results = sqlExecutor.execute(
            "select ss_store_sk, sum(ss_sales_price)  from test_ss40g_no_continuous_1_column_1_model where ss_coupon_amt='103.67'  group by ss_store_sk"
        )
        sqlExecutor.execute("drop table test_ss40g_no_continuous_1_column_1_model")
        self.assertFalse(results.empty)

    def test_no_continuous_categorical1_one_model_stratified(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set n_epoch=2")
        sqlExecutor.execute("set encoder='embedding'")
        sqlExecutor.execute("set one_model='true'")
        sqlExecutor.execute(
            "set table_header="
            + "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|"
            + "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|"
            + "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|"
            + "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|"
            + "ss_net_paid_inc_tax|ss_net_profit|none'"
        )
        sqlExecutor.execute(
            "drop table test_ss40g_no_continuous_1_column_1_model_stratified"
        )
        sqlExecutor.execute(
            "create table test_ss40g_no_continuous_1_column_1_model_stratified(ss_sales_price real, ss_coupon_amt categorical) from 'data/tpcds/10g/ss_10g_100.csv' GROUP BY ss_store_sk method stratified size 100"
        )
        results = sqlExecutor.execute(
            "select ss_store_sk, sum(ss_sales_price)  from test_ss40g_no_continuous_1_column_1_model_stratified where ss_coupon_amt='103.67'  group by ss_store_sk"
        )
        sqlExecutor.execute(
            "drop table test_ss40g_no_continuous_1_column_1_model_stratified"
        )
        self.assertFalse(results.empty)

    def test_no_continuous_categorical2_one_model_stratified(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set n_epoch=2")
        sqlExecutor.execute("set encoder='embedding'")
        sqlExecutor.execute("set one_model='true'")
        sqlExecutor.execute(
            "set table_header="
            + "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|"
            + "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|"
            + "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|"
            + "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|"
            + "ss_net_paid_inc_tax|ss_net_profit|none'"
        )
        sqlExecutor.execute(
            "drop table test_ss40g_no_continuous_1_column_1_model_stratified"
        )
        sqlExecutor.execute(
            "create table test_ss40g_no_continuous_1_column_1_model_stratified(ss_sales_price real, ss_coupon_amt categorical, ss_customer_sk categorical) from 'data/tpcds/10g/ss_10g_100.csv' GROUP BY ss_store_sk method uniform size 100"
        )
        results = sqlExecutor.execute(
            "select ss_store_sk, sum(ss_sales_price)  from test_ss40g_no_continuous_1_column_1_model_stratified where ss_coupon_amt='103.67' and ss_customer_sk='415915' group by ss_store_sk"
        )
        sqlExecutor.execute(
            "drop table test_ss40g_no_continuous_1_column_1_model_stratified"
        )
        self.assertFalse(results.empty)

    def test_plot(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set n_epoch=10")
        sqlExecutor.execute("set encoder='embedding'")
        sqlExecutor.execute("set one_model='true'")
        sqlExecutor.execute("set plot='True'")
        sqlExecutor.execute(
            "set table_header="
            + "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|"
            + "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|"
            + "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|"
            + "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|"
            + "ss_net_paid_inc_tax|ss_net_profit|none'"
        )
        sqlExecutor.execute("drop table test_plot")
        sqlExecutor.execute(
            "create table test_plot(ss_sales_price real, ss_sold_date_sk real) from '/home/quincy/Documents/workspace/data/tpcds/100g/ss_100g_2m.csv' GROUP BY ss_store_sk method uniform size 10000"  #data/tpcds/10g/ss_10g_100.csv
        )
        results = sqlExecutor.execute(
            "select ss_store_sk, sum(ss_sales_price)  from test_plot where 2451119  <=ss_sold_date_sk<= 2451483  group by ss_store_sk"
        )
        sqlExecutor.execute("drop table test_plot")
        self.assertFalse(results.empty)


if __name__ == "__main__":
    # unittest.main()
    # TestTpcDs().test_groupbys_range_no_categorical_gb1()
    # TestTpcDs().test_groupbys_range_no_categorical_gb2()
    # TestTpcDs().test_groupbys_range_no_categorical_gb1_stratified()
    # TestTpcDs().test_groupbys_range_no_categorical_gb2_stratified()
    # TestTpcDs().test_groupbys_range_no_categorical_gb2_stratified_sample_only()
    # TestTpcDs().test_categorical_one_model()
    # TestTpcDs().test_categorical_one_model_stratified()
    # TestTpcDs().test_embedding()
    # TestTpcDs().test_gogs_categorical()
    # TestTpcDs().test_no_continuous1()
    # TestTpcDs().test_no_continuous_categorical_1()
    # TestTpcDs().test_no_continuous_categorical_2()
    # TestTpcDs().test_no_continuous_categorical_one_model_uniform()
    # TestTpcDs().test_no_continuous_categorical1_one_model_stratified()
    # TestTpcDs().test_no_continuous_categorical2_one_model_stratified()
    TestTpcDs().test_plot()