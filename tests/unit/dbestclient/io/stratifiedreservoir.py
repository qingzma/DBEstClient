# Created by Qingzhi Ma at 2020-11-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
import unittest

from dbestclient.io.stratifiedreservoir import StratifiedReservoir


class TestStratifiedReservoir(unittest.TestCase):
    """"""

    # def test_tpcds_1job(self):
    #     sr = StratifiedReservoir(
    #         "data/tpcds/40G/ss_1k.csv",
    #         file_header="ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|ss_net_paid_inc_tax|ss_net_profit|none",
    #         n_jobs=1,
    #         capacity=5,
    #     )
    #     sr.make_sample(
    #         gb_cols=["ss_store_sk"],
    #         equality_cols=["ss_coupon_amt"],
    #         feature_cols=["ss_sold_date_sk", "ss_ext_wholesale_cost"],
    #         label_cols=["ss_sales_price"],
    #         split_char="|",
    #     )

    #     self.assertEqual(sr.size(), 1000)

    # def test_tpcds_2job(self):
    #     sr = StratifiedReservoir(
    #         "data/tpcds/40G/ss_1k.csv",
    #         file_header="ss_sold_date_sk|ss_sold_time_sk|ss_item_sk|ss_customer_sk|ss_cdemo_sk|ss_hdemo_sk|ss_addr_sk|ss_store_sk|ss_promo_sk|ss_ticket_number|ss_quantity|ss_wholesale_cost|ss_list_price|ss_sales_price|ss_ext_discount_amt|ss_ext_sales_price|ss_ext_wholesale_cost|ss_ext_list_price|ss_ext_tax|ss_coupon_amt|ss_net_paid|ss_net_paid_inc_tax|ss_net_profit|none",
    #         n_jobs=2,
    #         capacity=5,
    #     )
    #     sr.make_sample(
    #         gb_cols=["ss_store_sk"],
    #         equality_cols=["ss_coupon_amt"],
    #         feature_cols=["ss_sold_date_sk", "ss_ext_wholesale_cost"],
    #         label_cols=["ss_sales_price"],
    #         split_char="|",
    #     )

    #     self.assertEqual(sr.size(), 1000)

    # def test_toy_no_header_1(self):
    #     sr = StratifiedReservoir("data/toy/toy.txt",
    #                              file_header="range1,range2,cate1,cate2,gb1,gb2,label",
    #                              n_jobs=1, capacity=5)
    #     sr.make_sample(gb_cols=["gb1", "gb2"], equality_cols=["cate1", "cate2"], feature_cols=["range1", "range2"],
    #                    label_cols=["label"], split_char=',')
    #     self.assertEqual(sr.sample, {'store_id1,cust_id1,london,male': [['store_id1', 'cust_id1', 'london', 'male', '1.0', '2.0', 'salary1'], [
    #                      'store_id1', 'cust_id1', 'london', 'male', '1.1', '2.1', 'salary2']],
    #         'store_id1,cust_id2,paris,male': [['store_id1', 'cust_id2', 'paris', 'male', '1.1', '2.1', 'salary3']]})

    # def test_toy_no_header_2(self):
    #     sr = StratifiedReservoir("data/toy/toy.txt",
    #                              file_header="range1,range2,cate1,cate2,gb1,gb2,label",
    #                              n_jobs=1, capacity=5)
    #     sr.make_sample(gb_cols=["gb1", "gb2"], equality_cols=["cate1", "cate2"], feature_cols=["range1", "range2"],
    #                    label_cols=["label"], split_char=',')
    #     cate, features, labels = sr.get_categorical_features_label()
    #     self.assertEqual([cate, features, labels], [{'store_id1,cust_id1,london,male': [['store_id1', 'cust_id1', 'london', 'male'], ['store_id1', 'cust_id1', 'london', 'male']], 'store_id1,cust_id2,paris,male': [['store_id1', 'cust_id2', 'paris', 'male']]},
    #                                                 {'store_id1,cust_id1,london,male': [['1.0', '2.0'], [
    #                                                     '1.1', '2.1']], 'store_id1,cust_id2,paris,male': [['1.1', '2.1']]},
    #                                                 {'store_id1,cust_id1,london,male': [
    #                                                     'salary1', 'salary2'], 'store_id1,cust_id2,paris,male': ['salary3']}
    #                                                 ])

    # def test_toy_with_header_1job(self):
    #     sr = StratifiedReservoir("data/toy/toy_with_header.txt",
    #                              n_jobs=1, capacity=5)
    #     sr.make_sample(gb_cols=["gb1", "gb2"], equality_cols=["cate1", "cate2"], feature_cols=["range1", "range2"],
    #                    label_cols=["label"], split_char=',')
    #     # print(sr.sample, "-"*80)
    #     self.assertEqual(sr.sample, {'store_id1,cust_id1,london,male': [['store_id1', 'cust_id1', 'london', 'male', '1.0', '2.0', 'salary1'], [
    #                      'store_id1', 'cust_id1', 'london', 'male', '1.1', '2.1', 'salary2']],
    #         'store_id1,cust_id2,paris,male': [['store_id1', 'cust_id2', 'paris', 'male', '1.1', '2.1', 'salary3']]})

    # def test_toy_with_header_2job(self):
    #     sr = StratifiedReservoir("data/toy/toy_with_header.txt",
    #                              n_jobs=2, capacity=5)
    #     sr.make_sample(gb_cols=["gb1", "gb2"], equality_cols=["cate1", "cate2"], feature_cols=["range1", "range2"],
    #                    label_cols=["label"], split_char=',')
    #     # print(sr.sample, "-"*80)
    #     self.assertEqual(sr.sample, {'store_id1,cust_id1,london,male': [['store_id1', 'cust_id1', 'london', 'male', '1.0', '2.0', 'salary1'], [
    #                      'store_id1', 'cust_id1', 'london', 'male', '1.1', '2.1', 'salary2']],
    #         'store_id1,cust_id2,paris,male': [['store_id1', 'cust_id2', 'paris', 'male', '1.1', '2.1', 'salary3']]})

    # def test_hw(self):
    #     sr = StratifiedReservoir(
    #         "../data/huawei/merged",
    #         file_header="ts,apmac,accType,radioid,band,ssid,usermac,downSpeed,rssi,upLinkSpeed,downLinkSpeed,txDiscardRatio,latency,downBytes,upBytes,kpiCount,authTimeoutTimes,assoFailTimes,authFailTimes,dhcpFailTimes,assoSuccTimes,authSuccTimes,dhcpSuccTimes,dot1XSuccTimes,dot1XFailTimes,onlineSuccTimes,txDiscardFrames,txFrames,tenantId,siteId,siteName,directRegion,regionLevelOne,regionLevelTwo,regionLevelThree,regionLevelFour,regionLevelFive,regionLevelSix,regionLevelSeven,regionLevelEight,parentResId,acName,resId,apname,publicArea,vendor,duration,badCount,badTime,lowRssiCount,lowRssiDur,highLatencyCount,highLatencyDur,highDiscardCount,highDiscardDur,nonFiveGCount,nonFiveGDur,exception_flag,last_acc_rst,linkQuality,portal_succ_times,portal_fail_times,roam_succ_times,roam_fail_times",
    #         n_jobs=8,
    #         capacity=100,
    #     )
    #     sr.make_sample(
    #         gb_cols=["ts"],
    #         equality_cols=["regionLevelEight", "ssid"],
    #         feature_cols=["downSpeed"],
    #         label_cols=["latency"],
    #         split_char=",",
    #     )

    #     ft = sr.get_ft()
    #     # for key in ft:
    #     #     print(key, ft[key])
    #     # print("predictions", predictions)
    #     self.assertEqual(81526479, sr.size())


if __name__ == "__main__":
    unittest.main()
    # TestStratifiedReservoir().test_tpcds_1job()
    # TestStratifiedReservoir().test_tpcds_2job()
    # TestStratifiedReservoir().test_toy_no_header_2()
    # TestStratifiedReservoir().test_toy_with_header_1job()
    # TestStratifiedReservoir().test_toy_with_header_2job()
    # TestStratifiedReservoir().test_hw()
