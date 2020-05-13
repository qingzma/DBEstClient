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

from dbestclient.parser.parser import DBEstParser


class TestParser(unittest.TestCase):
    """Test the parser
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = DBEstParser()
        self.q_set = "set a = 5;"
        self.q_ddl = "create table ss40g_1(ss_sales_price real, ss_sold_date_sk real, ss_coupon_amt categorical) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk method uniform size 60000 scale data num_of_points2.csv"
        self.q_dml = "select avg(ss_sales_price)  from ss40g_1 where ss_sold_date_sk between 2451119  and 2451483 and ss_coupon_amt=''  group by ss_store_sk"
        self.sqls = [self.q_set, self.q_ddl, self.q_dml]

    def test_query_type_set(self):
        types = []

        for sql in self.sqls:
            self.parser.parse(sql)
            types.append(self.parser.get_query_type())
        self.assertEqual(types, ["set", "ddl", "dml"])

    def test_query_type_ddl(self):
        sql = "create table ss40g_1(ss_sales_price real, ss_sold_date_sk real, ss_coupon_amt categorical) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk method uniform size 60000 scale data num_of_points2.csv"
        self.assertEqual(1, 1)

    def test_query_type_dml(self):
        sql = "select avg(ss_sales_price)  from ss40g_1 where ss_sold_date_sk between 2451119  and 2451483 and ss_coupon_amt=''  group by ss_store_sk"
        self.assertEqual(1, 1)

    def test_ddl(self):
        """Test ddl
        """
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
