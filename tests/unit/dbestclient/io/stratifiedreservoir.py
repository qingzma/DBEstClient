# Created by Qingzhi Ma at 2020-11-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
import unittest

from dbestclient.io.stratifiedreservoir import StratifiedReservoir


class TestStratifiedReservoir(unittest.TestCase):
    '''
    '''

    def test_headers(self):
        sr = StratifiedReservoir("data/tpcds/40G/ss_600k.csv",
                                 file_header="ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|ss_net_paid_inc_tax|ss_net_profit|none",
                                 n_jobs=1, capacity=5)
        sr.make_sample(gb_cols=["ss_store_sk"], equality_cols=["ss_coupon_amt"], feature_cols=["ss_sold_date_sk", "ss_ext_wholesale_cost"],
                       label_cols=["ss_sales_price"], split_char='|')

        # print("predictions", predictions)
        self.assertTrue(1 == 1)


if __name__ == "__main__":
    TestStratifiedReservoir().test_headers()
