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

'''
Table: ci_campusclient_clientstat_5m
Number of rows: 81526479
'''

def run():
    sqlExecutor = SqlExecutor()

    build_models(sqlExecutor)
    query(sqlExecutor,"hw_q1_1pct")
    query(sqlExecutor,"hw_q1_5pct")
    query(sqlExecutor,"hw_q1_10pct")
    query(sqlExecutor,"hw_q1_20pct")


def build_models(sqlExecutor):
    # sqlExecutor.execute("drop table hw_q1_test")

    sqlExecutor.execute("set b_grid_search='False'")
    # sqlExecutor.execute("set csv_split_char=','")
    sqlExecutor.execute("set encoder='embedding'")
    sqlExecutor.execute("set table_header=" +
                        "'ts,apmac,acctype,radioid,band,ssid,usermac,downspeed,rssi,uplinkspeed,downlinkspeed,txdiscardratio,latency,downbytes,upbytes,kpicount,authtimeouttimes,assofailtimes,authfailtimes,dhcpfailtimes,assosucctimes,authsucctimes,dhcpsucctimes,dot1xsucctimes,dot1xfailtimes,onlinesucctimes,txdiscardframes,txframes,tenantid,siteid,sitename,directregion,regionlevelone,regionleveltwo,regionlevelthree,regionlevelfour,regionlevelfive,regionlevelsix,regionlevelseven,regionleveleight,parentresid,acname,resid,apname,publicarea,vendor,duration,badcount,badtime,lowrssicount,lowrssidur,highlatencycount,highlatencydur,highdiscardcount,highdiscarddur,nonfivegcount,nonfivegdur,exception_flag,last_acc_rst,linkquality,portal_succ_times,portal_fail_times,roam_succ_times,roam_fail_times'")
    sqlExecutor.execute("create table hw_q1_test(usermac categorical distinct, ts real,tenantId categorical, ssid  categorical,kpiCount categorical,regionLevelEight categorical)  "  #
                        "FROM '../data/huawei/sample.csv' "
                        "GROUP BY ts "
                        "method uniform "
                        "size  11800 ")  
    
    sqlExecutor.execute("set table_header=" +
                        "'ts,apmac,acctype,radioid,band,ssid,usermac,downspeed,rssi,uplinkspeed,downlinkspeed,txdiscardratio,latency,downbytes,upbytes,kpicount,authtimeouttimes,assofailtimes,authfailtimes,dhcpfailtimes,assosucctimes,authsucctimes,dhcpsucctimes,dot1xsucctimes,dot1xfailtimes,onlinesucctimes,txdiscardframes,txframes,tenantid,siteid,sitename,directregion,regionlevelone,regionleveltwo,regionlevelthree,regionlevelfour,regionlevelfive,regionlevelsix,regionlevelseven,regionleveleight,parentresid,acname,resid,apname,publicarea,vendor,duration,badcount,badtime,lowrssicount,lowrssidur,highlatencycount,highlatencydur,highdiscardcount,highdiscarddur,nonfivegcount,nonfivegdur,exception_flag,last_acc_rst,linkquality,portal_succ_times,portal_fail_times,roam_succ_times,roam_fail_times'")
    sqlExecutor.execute("create table hw_q1_1pct(usermac categorical distinct, ts real,tenantId categorical, ssid  categorical,kpiCount categorical,regionLevelEight categorical)  "  #
                        "FROM '../data/huawei/percent1.csv' "
                        "GROUP BY ts "
                        "method uniform "
                        "size  0.01 ")  
    
    sqlExecutor.execute("set table_header=" +
                        "'ts,apmac,acctype,radioid,band,ssid,usermac,downspeed,rssi,uplinkspeed,downlinkspeed,txdiscardratio,latency,downbytes,upbytes,kpicount,authtimeouttimes,assofailtimes,authfailtimes,dhcpfailtimes,assosucctimes,authsucctimes,dhcpsucctimes,dot1xsucctimes,dot1xfailtimes,onlinesucctimes,txdiscardframes,txframes,tenantid,siteid,sitename,directregion,regionlevelone,regionleveltwo,regionlevelthree,regionlevelfour,regionlevelfive,regionlevelsix,regionlevelseven,regionleveleight,parentresid,acname,resid,apname,publicarea,vendor,duration,badcount,badtime,lowrssicount,lowrssidur,highlatencycount,highlatencydur,highdiscardcount,highdiscarddur,nonfivegcount,nonfivegdur,exception_flag,last_acc_rst,linkquality,portal_succ_times,portal_fail_times,roam_succ_times,roam_fail_times'")
    sqlExecutor.execute("create table hw_q1_5pct(usermac categorical distinct, ts real,tenantId categorical, ssid  categorical,kpiCount categorical,regionLevelEight categorical)  "  #
                        "FROM '../data/huawei/4_075m_5_percent.csv' "
                        "GROUP BY ts "
                        "method uniform "
                        "size  0.049983760490871 ") 
    
    sqlExecutor.execute("set table_header=" +
                        "'ts,apmac,acctype,radioid,band,ssid,usermac,downspeed,rssi,uplinkspeed,downlinkspeed,txdiscardratio,latency,downbytes,upbytes,kpicount,authtimeouttimes,assofailtimes,authfailtimes,dhcpfailtimes,assosucctimes,authsucctimes,dhcpsucctimes,dot1xsucctimes,dot1xfailtimes,onlinesucctimes,txdiscardframes,txframes,tenantid,siteid,sitename,directregion,regionlevelone,regionleveltwo,regionlevelthree,regionlevelfour,regionlevelfive,regionlevelsix,regionlevelseven,regionleveleight,parentresid,acname,resid,apname,publicarea,vendor,duration,badcount,badtime,lowrssicount,lowrssidur,highlatencycount,highlatencydur,highdiscardcount,highdiscarddur,nonfivegcount,nonfivegdur,exception_flag,last_acc_rst,linkquality,portal_succ_times,portal_fail_times,roam_succ_times,roam_fail_times'")
    sqlExecutor.execute("create table hw_q1_10pct(usermac categorical distinct, ts real,tenantId categorical, ssid  categorical,kpiCount categorical,regionLevelEight categorical)  "  #
                        "FROM '../data/huawei/8_1m_10_percent.csv' "
                        "GROUP BY ts "
                        "method uniform "
                        "size  0.099354223307007 ")  
    
    sqlExecutor.execute("set table_header=" +
                        "'ts,apmac,acctype,radioid,band,ssid,usermac,downspeed,rssi,uplinkspeed,downlinkspeed,txdiscardratio,latency,downbytes,upbytes,kpicount,authtimeouttimes,assofailtimes,authfailtimes,dhcpfailtimes,assosucctimes,authsucctimes,dhcpsucctimes,dot1xsucctimes,dot1xfailtimes,onlinesucctimes,txdiscardframes,txframes,tenantid,siteid,sitename,directregion,regionlevelone,regionleveltwo,regionlevelthree,regionlevelfour,regionlevelfive,regionlevelsix,regionlevelseven,regionleveleight,parentresid,acname,resid,apname,publicarea,vendor,duration,badcount,badtime,lowrssicount,lowrssidur,highlatencycount,highlatencydur,highdiscardcount,highdiscarddur,nonfivegcount,nonfivegdur,exception_flag,last_acc_rst,linkquality,portal_succ_times,portal_fail_times,roam_succ_times,roam_fail_times'")
    sqlExecutor.execute("create table hw_q1_20pct(usermac categorical distinct, ts real,tenantId categorical, ssid  categorical,kpiCount categorical,regionLevelEight categorical)  "  #
                        "FROM '../data/huawei/16_2m_20_percent.csv' "
                        "GROUP BY ts "
                        "method uniform "
                        "size  0.198708446614014 ")  
    
    sqlExecutor.execute("set table_header=" +
                        "'ts,apmac,acctype,radioid,band,ssid,usermac,downspeed,rssi,uplinkspeed,downlinkspeed,txdiscardratio,latency,downbytes,upbytes,kpicount,authtimeouttimes,assofailtimes,authfailtimes,dhcpfailtimes,assosucctimes,authsucctimes,dhcpsucctimes,dot1xsucctimes,dot1xfailtimes,onlinesucctimes,txdiscardframes,txframes,tenantid,siteid,sitename,directregion,regionlevelone,regionleveltwo,regionlevelthree,regionlevelfour,regionlevelfive,regionlevelsix,regionlevelseven,regionleveleight,parentresid,acname,resid,apname,publicarea,vendor,duration,badcount,badtime,lowrssicount,lowrssidur,highlatencycount,highlatencydur,highdiscardcount,highdiscarddur,nonfivegcount,nonfivegdur,exception_flag,last_acc_rst,linkquality,portal_succ_times,portal_fail_times,roam_succ_times,roam_fail_times'")
    sqlExecutor.execute("create table hw_q1_1pct_no_distinct(usermac categorical, ts real,tenantId categorical, ssid  categorical,kpiCount categorical,regionLevelEight categorical)  "  #
                        "FROM '../data/huawei/percent1.csv' "
                        "GROUP BY ts "
                        "method uniform "
                        "size  0.01 ") 


def query(sqlExecutor, mdl_name):
    ### query 1 day
    # sqlExecutor.execute("select ts, count(distinct usermac) from  " + mdl_name + " "+
    #                     "where   unix_timestamp('2020-03-05T12:00:00.000Z') <=ts<= unix_timestamp('2020-03-06T12:00:00.000Z') "
    #                     "AND tenantId = 'default-organization-id' "
    #                     "AND ssid = 'Tencent' "
    #                     "AND kpiCount >=1  "
    #                     "AND regionLevelEight='3151a52f-0755-11ea-840e-60def3781da5'"
    #                     "GROUP BY ts")

    ### query another day
    sqlExecutor.execute("select ts, count(distinct usermac) from  " + mdl_name + " "+
                        "where   unix_timestamp('2020-03-05T16:00:00.000Z') <=ts<= unix_timestamp('2020-03-06T16:00:00.000Z') "
                        "AND tenantId = 'default-organization-id' "
                        "AND ssid = 'Huawei' "
                        "AND kpiCount >=1  "
                        "AND regionLevelEight='287d4300-06bb-11ea-840e-60def3781da5'"
                        "GROUP BY ts")

    # 522459f1-0755-11ea-840e-60def3781da5      287d4300-06bb-11ea-840e-60def3781da5


if __name__ == "__main__":
    run()
