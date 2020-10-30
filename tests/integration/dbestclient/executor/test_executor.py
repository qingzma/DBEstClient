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

    def test_groupbys(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set encoder='binary'")
        sqlExecutor.execute("set table_header=" +
                            "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
                            "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
                            "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
                            "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
                            "ss_net_paid_inc_tax|ss_net_profit|none'"
                            )
        sqlExecutor.execute(
            "create table test_ss40g_groupbys(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk,ss_coupon_amt method uniform size 600")
        predictions = sqlExecutor.execute(
            "select avg(ss_sales_price)  from test_ss40g_groupbys where   2451119  <=ss_sold_date_sk<= 2451483  group by ss_store_sk,ss_coupon_amt")
        sqlExecutor.execute("drop table test_ss40g_groupbys")
        self.assertTrue(predictions)

    def test_categorical(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set table_header=" +
                            "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
                            "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
                            "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
                            "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
                            "ss_net_paid_inc_tax|ss_net_profit|none'"
                            )
        sqlExecutor.execute(
            "create table test_ss40g_categorical(ss_sales_price real, ss_sold_date_sk real, ss_coupon_amt categorical) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk method uniform size 600")
        predictions = sqlExecutor.execute(
            "select avg(ss_sales_price)  from test_ss40g_categorical where   2451119  <=ss_sold_date_sk<= 2451483 and ss_coupon_amt=''  group by ss_store_sk")
        sqlExecutor.execute("drop table test_ss40g_categorical")
        self.assertTrue(predictions)

    def test_categorical_one_model(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set one_model='true'")
        sqlExecutor.execute("set table_header=" +
                            "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
                            "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
                            "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
                            "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
                            "ss_net_paid_inc_tax|ss_net_profit|none'"
                            )
        sqlExecutor.execute(
            "create table test_ss40g_categorical_one_model(ss_sales_price real, ss_sold_date_sk real,  ss_coupon_amt categorical, ) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk method uniform size 600")  # ss_ext_discount_amt categorical
        predictions = sqlExecutor.execute(
            "select count(ss_sales_price)  from test_ss40g_categorical_one_model where   2451119  <=ss_sold_date_sk<= 2451483 and ss_coupon_amt='0.00'   group by ss_store_sk")  # and ss_ext_discount_amt='0.00'
        # sqlExecutor.execute("drop table test_ss40g_categorical_one_model")
        self.assertTrue(predictions)

    def test_gogs_no_categorical(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set b_use_gg='True'")
        sqlExecutor.execute("set n_per_gg=10")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set table_header=" +
                            "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
                            "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
                            "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
                            "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
                            "ss_net_paid_inc_tax|ss_net_profit|none'"
                            )
        sqlExecutor.execute(
            "create table test_ss40g_gogs_no_categorical(ss_sales_price real, ss_sold_date_sk real) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk method uniform size 600")
        predictions = sqlExecutor.execute(
            "select avg(ss_sales_price)  from test_ss40g_gogs_no_categorical where   2451119  <=ss_sold_date_sk<= 2451483  group by ss_store_sk")
        sqlExecutor.execute("drop table test_ss40g_gogs_no_categorical")
        self.assertTrue(predictions)

    def test_gogs_categorical(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set b_use_gg='True'")
        sqlExecutor.execute("set n_per_gg=10")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set table_header=" +
                            "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
                            "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
                            "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
                            "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
                            "ss_net_paid_inc_tax|ss_net_profit|none'"
                            )
        sqlExecutor.execute(
            "create table test_ss40g_gogs_categorical(ss_sales_price real, ss_sold_date_sk real, ss_coupon_amt categorical) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk method uniform size 600")
        predictions = sqlExecutor.execute(
            "select avg(ss_sales_price)  from test_ss40g_gogs_categorical where   2451119  <=ss_sold_date_sk<= 2451483 and ss_coupon_amt='' group by ss_store_sk")
        sqlExecutor.execute("drop table test_ss40g_gogs_categorical")
        self.assertTrue(predictions)

    def test_drop_clause(self):
        sqlExecutor = SqlExecutor()
        status = sqlExecutor.execute("drop table model2drop")
        self.assertFalse(status)

    def test_embedding(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set b_grid_search='False'")
        sqlExecutor.execute("set csv_split_char='|'")
        sqlExecutor.execute("set encoder='embedding'")
        sqlExecutor.execute("set table_header=" +
                            "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
                            "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
                            "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
                            "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
                            "ss_net_paid_inc_tax|ss_net_profit|none'"
                            )
        sqlExecutor.execute(
            "create table test_ss40g_embedding(ss_sales_price real, ss_sold_date_sk real) from '../data/tpcds/10g/ss_10g_520k.csv' GROUP BY ss_store_sk method uniform size 600")
        predictions = sqlExecutor.execute(
            "select avg(ss_sales_price)  from test_ss40g_embedding where   2451119  <=ss_sold_date_sk<= 2451483  group by ss_store_sk")
        sqlExecutor.execute("drop table test_ss40g_embedding")
        self.assertTrue(predictions)

    # def test_no_continuous1(self):
    #     sqlExecutor = SqlExecutor()
    #     sqlExecutor.execute("set b_grid_search='False'")
    #     sqlExecutor.execute("set csv_split_char='|'")
    #     sqlExecutor.execute("set encoder='binary'")
    #     sqlExecutor.execute("set table_header=" +
    #                         "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
    #                         "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
    #                         "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
    #                         "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
    #                         "ss_net_paid_inc_tax|ss_net_profit|none'"
    #                         )
    #     sqlExecutor.execute(
    #         "create table test_ss40g_no_continuous1(ss_sales_price real) from '../data/tpcds/10g/ss_10g_520k.csv' GROUP BY ss_store_sk method uniform size 600")  # , ss_coupon_amt categorical
    #     sqlExecutor.execute("drop table test_ss40g_no_continuous1")
    #     self.assertTrue(True)

    # def test_no_continuous2(self):
    #     sqlExecutor = SqlExecutor()
    #     sqlExecutor.execute("set b_grid_search='False'")
    #     sqlExecutor.execute("set csv_split_char='|'")
    #     sqlExecutor.execute("set encoder='binary'")
    #     sqlExecutor.execute("set table_header=" +
    #                         "'ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|" +
    #                         "ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|" +
    #                         "ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|" +
    #                         "ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|" +
    #                         "ss_net_paid_inc_tax|ss_net_profit|none'"
    #                         )
    #     sqlExecutor.execute(
    #         "create table test_ss40g_no_continuous2(ss_sales_price real, ss_coupon_amt categorical) from '../data/tpcds/10g/ss_10g_520k.csv' GROUP BY ss_store_sk method uniform size 600")
    #     sqlExecutor.execute("drop table test_ss40g_no_continuous2")
    #     self.assertTrue(True)


class TestHw(unittest.TestCase):
    def test_cpu(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("create table test_hw(usermac categorical , ts real,tenantId categorical, ssid  categorical,kpiCount categorical,regionLevelEight categorical)  "  #
                            "FROM '/home/quincy/Documents/workspace/DBEstClient/tests/integration/fixtures/sample_1k.csv' "
                            "GROUP BY ts "
                            "method uniform "
                            "size  1000 "  # 118567, 81526479
                            "scale data;")
        predictions = sqlExecutor.execute("select ts, count(usermac) from test_hw "
                                          "where   unix_timestamp('2020-02-05T12:00:00.000Z') <=ts<= unix_timestamp('2020-04-06T12:00:00.000Z') "
                                          "AND tenantId = 'default-organization-id' "
                                          "AND ssid = 'Tencent' "
                                          "AND kpiCount >=1  "
                                          "AND regionLevelEight='287d4300-06bb-11ea-840e-60def3781da5'"
                                          "GROUP BY ts;")
        sqlExecutor.execute("drop table test_hw")
        # print("predictions", predictions)
        self.assertTrue(abs(predictions['1583402400000']-316.683) < 10)


if __name__ == "__main__":
    # unittest.main()
    # TestTpcDs().test_groupbys()
    # TestTpcDs().test_categorical_one_model()
    # TestHw().test_cpu()
    # TestTpcDs().test_embedding()
    TestTpcDs().test_no_continuous1()
    TestTpcDs().test_no_continuous2()
