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

from dbestclient.parser.parser import (
    DBEstParser,
    parse_usecols_check_shared_attributes_exist,
    parse_y_check_need_ft_only,
)


class TestParser(unittest.TestCase):
    """Test the parser"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = DBEstParser()
        self.q_set = "set a = 5;"
        self.q_create = "create table ss40g_1(ss_sales_price real, ss_sold_date_sk real, ss_coupon_amt categorical) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk method uniform size 60000 scale data num_of_points2.csv"
        self.q_select = "select avg(ss_sales_price)  from ss40g_1 where   2451119 <= ss_sold_date_sk<= 2451483 and ss_coupon_amt=''  group by ss_store_sk"
        self.q_drop = "drop table haha"
        self.sqls = [self.q_set, self.q_create, self.q_select, self.q_drop]

    def test_query_type_set(self):
        types = []

        for sql in self.sqls:
            self.parser.parse(sql)
            types.append(self.parser.get_query_type())
        self.assertEqual(types, ["set", "create", "select", "drop"])

    # def test_query_type_create(self):
    #     sql = "create table ss40g_1(ss_sales_price real, ss_sold_date_sk real, ss_coupon_amt categorical) from '/data/tpcds/40G/ss_600k.csv' GROUP BY ss_store_sk method uniform size 60000 scale data num_of_points2.csv"
    #     self.assertEqual(1, 1)

    # def test_query_type_select(self):
    #     sql = "select avg(ss_sales_price)  from ss40g_1 where ss_sold_date_sk between 2451119  and 2451483 and ss_coupon_amt=''  group by ss_store_sk"
    #     self.assertEqual(1, 1)

    def test_drop_get_model(self):
        """Test DROP query"""
        sql = "drop table model2drop"
        self.parser.parse(sql)
        sql_type = self.parser.drop_get_model()
        self.assertEqual(sql_type, "model2drop")

    def test_get_sampling_ratio(self):
        """test get_sampling_ratio()"""

        sql1 = "create table test(y float, x float) from '/path/to/data/file.csv' group by g method uniform size 0.1"
        sql2 = "create table test(y float, x float) from '/path/to/data/file.csv' group by g method uniform size 1000"
        sql3 = "create table test(y float, x float) from '/path/to/data/file.csv' group by g method uniform size '/data/file/haha.csv'"
        sqls = [sql1, sql2, sql3]
        sizes = []
        for sql in sqls:
            self.parser.parse(sql)
            sizes.append(self.parser.get_sampling_ratio())
        self.assertEqual(sizes, [0.1, 1000, "/data/file/haha.csv"])

    def test_get_dml_aggregate_function_and_variable(self):
        sql1 = "select z, var ( y ) from t_m where  unix_timestamp('2019-02-28T16:00:00.000Z')<=x <=unix_timestamp('2019-03-28T16:00:00.000Z') and 321<X1 < 1123 and x2 = 'HaHaHa' and x3='' and x4<5 GROUP BY z1, z2 ,x method uniform scale data   haha/num.csv  size 23"
        sql2 = "select z, count ( y ) from t_m where   x2 = 'HaHaHa' and x3='' and  GROUP BY z1, z2 ,x method uniform scale data   haha/num.csv  size 23"
        sqls = [sql1, sql2]
        sizes = []
        for sql in sqls:
            self.parser.parse(sql)
            sizes.append(list(self.parser.get_dml_aggregate_function_and_variable()))

        self.assertEqual(
            sizes, [["z", ["var", "y", None]], ["z", ["count", "y", None]]]
        )

    def test_get_dml_where_categorical_equal_and_range(self):
        sql1 = "select z, var ( y ) from t_m where  unix_timestamp('2019-02-28T16:00:00.000Z')<=x <=unix_timestamp('2019-03-28T16:00:00.000Z') and 321<X1 < 1123 and x2 = 'HaHaHa' and x3='' and x4<5 GROUP BY z1, z2 ,x method uniform scale data   haha/num.csv  size 23"
        sql2 = "select z, count ( y ) from t_m where   x2 = 'HaHaHa' and x3='' and  GROUP BY z1, z2 ,x method uniform scale data   haha/num.csv  size 23"
        # sql3= "SELECT dest_state_abr, AVG( taxi_out ) FROM  tbl  where  1500 <=distance <= 2500 and  unique_carrier = 'UA'  GROUP BY dest_state_abr"
        sqls = [sql1, sql2]
        sizes = []
        for sql in sqls:
            self.parser.parse(sql)
            sizes.append(list(self.parser.get_dml_where_categorical_equal_and_range()))

        self.assertEqual(
            sizes,
            [
                [
                    ["x2", "x3"],
                    ["'HaHaHa'", "''"],
                    {
                        "x": [1551369600000, 1553788800000, False, True],
                        "X1": [321.0, 1123.0, False, False],
                        "x4": [None, "5", False, False],
                    },
                ],
                [["x2", "x3"], ["'HaHaHa'", "''"], {}],
            ],
        )

    def test_ddl_get_y(self):
        """get the attribute which is aggregated."""
        sql1 = "create table mdl(y categorical distinct, x0 real, x2 categorical, x3 categorical) from tbl group by z method uniform size '/data/haha.csv'"
        sql2 = "create table mdl(y categorical, x2 categorical) from tbl group by z method uniform size '/data/haha.csv'"
        sql3 = "create table mdl(y categorical distinct) from tbl group by z method uniform size '/data/haha.csv'"
        sqls = [sql1, sql2, sql3]
        sizes = []
        for sql in sqls:
            self.parser.parse(sql)
            sizes.append(list(self.parser.get_y()))
        #     print(list(self.parser.get_y()))
        # print("sizes", sizes)
        self.assertEqual(
            sizes,
            [
                ["y", "categorical", "distinct"],
                ["y", "categorical", None],
                ["y", "categorical", "distinct"],
            ],
        )

    def test_ddl_get_x(self):
        sql1 = "create table mdl(y categorical distinct, x0 real, x2 categorical, x3 categorical) from tbl group by z method uniform size '/data/haha.csv'"
        sql2 = "create table mdl(y categorical, x2 categorical) from tbl group by z method uniform size '/data/haha.csv'"
        sql3 = "create table mdl(y categorical distinct) from tbl group by z method uniform size '/data/haha.csv'"
        sqls = [sql1, sql2, sql3]
        sizes = []
        for sql in sqls:
            self.parser.parse(sql)
            sizes.append(list(self.parser.get_x()))
        #     print(list(self.parser.get_x()))
        # print("sizes", sizes)
        self.assertEqual(sizes, [[["x0"], ["x2", "x3"]], [[], ["x2"]], [[], []]])

    def test_parse_usecols_shared(self):
        usecols = usecols = {
            "y": ["usermac", "categorical", None],
            "x_continous": [],
            "x_categorical": ["ts", "tenantid"],
            "gb": ["ts"],
        }
        status, usecols = parse_usecols_check_shared_attributes_exist(usecols)
        self.assertTrue(status)
        self.assertEqual(
            usecols,
            {
                "y": ["usermac", "categorical", None],
                "x_continous": [],
                "x_categorical": ["tenantid"],
                "gb": ["ts"],
            },
        )

    def test_parse_usecols_not_shared(self):
        usecols = usecols = {
            "y": ["usermac", "categorical", None],
            "x_continous": [],
            "x_categorical": ["tenantid"],
            "gb": ["ts"],
        }
        status, usecols = parse_usecols_check_shared_attributes_exist(usecols)
        self.assertFalse(status)
        self.assertEqual(
            usecols,
            {
                "y": ["usermac", "categorical", None],
                "x_continous": [],
                "x_categorical": ["tenantid"],
                "gb": ["ts"],
            },
        )

    def test_parse_y_check_need_ft_only(self):
        usecols = usecols = {
            "y": ["usermac", "categorical", None],
            "x_continous": [],
            "x_categorical": ["tenantid"],
            "gb": ["ts"],
        }
        status = parse_y_check_need_ft_only(usecols)
        self.assertTrue(status)

        usecols = usecols = {
            "y": ["usermac", "real", None],
            "x_continous": [],
            "x_categorical": ["tenantid"],
            "gb": ["ts"],
        }
        status = parse_y_check_need_ft_only(usecols)
        self.assertFalse(status)

        usecols = usecols = {
            "y": ["usermac", "real", None],
            "x_continous": ["a"],
            "x_categorical": ["tenantid"],
            "gb": ["ts"],
        }
        status = parse_y_check_need_ft_only(usecols)
        self.assertFalse(status)

        usecols = usecols = {
            "y": ["usermac", "categorical", None],
            "x_continous": ["a"],
            "x_categorical": ["tenantid"],
            "gb": ["ts"],
        }
        status = parse_y_check_need_ft_only(usecols)
        self.assertFalse(status)


if __name__ == "__main__":
    unittest.main()
