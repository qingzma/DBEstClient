#
# Created by Qingzhi Ma on Mon May 04 2020
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


from dbestclient.executor.executor import SqlExecutor


def run():
    sqlExecutor = SqlExecutor()

    build_models(sqlExecutor)
    query(sqlExecutor)


def build_models(sqlExecutor):
    sqlExecutor.execute("create table huawei_p1(usermac categorical , ts real,tenantId categorical, ssid  categorical,kpiCount categorical,regionLevelEight categorical )  "  #
                        "FROM '/data/huawei/percent1.csv' "
                        # "WHERE  ts between 0 and 10 "
                        # "AND tenantId = 'default-organization-id' "
                        # "AND kpiCount = 0 "
                        # "AND ssid = 'Apple' "
                        # "AND regionLevelEight = '287d4300-06bb-11ea-840e-60def3781da5' "
                        "GROUP BY ts "
                        "method uniform "
                        "size  0.01 "  # 118567
                        "scale data;", device='gpu')


def query(sqlExecutor):
    sqlExecutor.execute("select ts, count(usermac) from huawei_p1 "
                        "where ts between to_timestamp('2020-01-28T16:00:00.000Z') and to_timestamp('2020-04-28T16:00:00.000Z') "
                        "AND tenantId = 'default-organization-id' "
                        "AND ssid = 'Tencent' "
                        "AND kpiCount >= 2 "
                        "AND regionLevelEight='287d4300-06bb-11ea-840e-60def3781da5'"


                        # "AND regionLevelEight = '9f642594-20c2-4ccb-8f5d-97d5f59a1e18' "
                        "GROUP BY ts;", n_jobs=1, device='gpu')


if __name__ == "__main__":
    run()
