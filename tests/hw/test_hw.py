#
# Created by Qingzhi Ma on Wed Nov 13 2020
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


class TestHw(unittest.TestCase):
    """
    Table: ci_campusclient_clientstat_5m
    Number of rows: 81526479
    """

    def test_q1(self):
        sqlExecutor = SqlExecutor()
        sqlExecutor.execute("set one_model='False'")
        sqlExecutor.execute("set b_print_to_screen='True'")
        sqlExecutor.execute("drop table test_hw")
        sqlExecutor.execute(
            "create table test_hw(usermac categorical , ts real,tenantId categorical, ssid  categorical,kpiCount categorical,regionLevelEight categorical)  "  #
            "FROM 'data/hw/sample_1k.csv' "
            "GROUP BY ts "
            "method uniform "
            "size  999"
        )  # 118567, 81526479;")
        predictions = sqlExecutor.execute(
            "select ts, count(usermac) from test_hw "
            "where   unix_timestamp('2020-02-05T12:00:00.000Z') <=ts<= unix_timestamp('2020-04-06T12:00:00.000Z') "
            "AND tenantId = 'default-organization-id' "
            "AND ssid = 'Tencent' "
            "AND kpiCount >=1  "
            "AND regionLevelEight='287d4300-06bb-11ea-840e-60def3781da5'"
            "GROUP BY ts;"
        )
        sqlExecutor.execute("drop table test_hw")
        # print("predictions", predictions)
        self.assertFalse(predictions.empty)


if __name__ == "__main__":
    TestHw().test_q1()