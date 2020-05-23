# Created by Qingzhi Ma at 2019-07-25
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
import argparse
import sys

from dbestclient.cli.prompt import DBEstPrompt
from dbestclient.executor.executor import SqlExecutor
from dbestclient.ml import mdn
from dbestclient.socket import app_client, app_server


def main():
    p = DBEstPrompt()
    p.cmdloop()


def cmd():
    print("Welcome to DBEst++")

    parser = argparse.ArgumentParser(
        description='Process the input for DBEst++.')
    # parser.add_argument('--foo', action='store')
    parser.add_argument('--pm25', action='store_true',
                        help="run pm25 experiments")
    args = parser.parse_args()

    if args.pm25:
        print("run pm25")
        mdn.test_pm25_3d()


def slave():

    if len(sys.argv) != 2:
        print("Usage:", " dbestslave ", "<local IP>:<port>")
        print("Abort.")
        sys.exit(1)
    splits = sys.argv[1].split(":")
    host = splits[0]
    port = int(splits[1])
    print("starting slave at local ", (host, port))
    print("Welcome to DBEst++")
    sqlExecutor = SqlExecutor()
    app_server.run(host, port, sqlExecutor)


def master():
    print("Welcome to DBEst++")
    if len(sys.argv) != 2:
        print("Usage:", " dbestmaster ",
              "<host:port>, <host:port>, ... (lists of slaves)")
        print("Abort.")
        sys.exit(1)
    splits = sys.argv[1].split(":")
    host = splits[0]
    port = int(splits[1])
    print("starting master, ready for connections...")
    query = {"mdl_name": "ss40g.dill", 'func': 'avg', 'x_lb': 2451119.0, 'x_ub': 2451483.0, 'x_categorical_conditions': [['ss_coupon_amt', 'ss_quantity'], ["''", "''"], {'ss_sold_date_sk': [2451119.0, 2451483.0, True, True]}], 'runtime_config': {'device': 'cpu', 'n_jobs': 1, 'v': True, 'b_show_latency': True, 'b_print_to_screen': False, 'result2file': None, 'b_use_integral': False, 'n_division': 20, 'epsabs': 10.0, 'epsrel': 0.1, 'limit': 30, 'model_suffix': '.dill'}, 'sub_group': ['44,37', '44,54', '44,57', '44,6', '44,73', '44,74', '44,92', '46,25', '46,30', '46,6', '46,60', '46,63', '46,65', '46,66', '46,81', '46,82', '46,85', '49,18', '49,2', '49,30', '49,31', '49,40', '49,56', '49,6', '49,64', '49,65', '49,71', '49,78', '49,96', '49,99', '50,35', '50,4', '50,47', '50,50', '50,53', '50,76', '50,86', '50,9', '52,1', '52,12', '52,13', '52,14', '52,16', '52,17', '52,5', '52,62', '52,64', '52,71', '52,83', '55,1', '55,24', '55,32', '55,55', '55,62', '55,7', '55,76', '55,79', '55,98', '56,24', '56,30', '56,31', '56,4', '56,67', '56,78', '56,91', '56,98', '58,14', '58,25', '58,78', '58,84', '58,96', '58,99', '61,22', '61,25', '61,3', '61,40', '61,47', '61,48', '61,51', '61,59', '61,65', '61,79', '61,80', '61,94', '62,15', '62,21', '62,36', '62,41', '62,54', '62,62', '62,73', '62,75', '62,84', '62,90', '62,91', '64,1', '64,11', '64,34', '64,4', '64,5', '64,50', '64,66', '64,78', '64,80', '64,88', '64,89', '67,12', '67,13', '67,17', '67,29', '67,31',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       '67,40', '67,49', '67,55', '67,57', '67,65', '67,67', '67,68', '67,70', '67,92', '68,12', '68,17', '68,21', '68,28', '68,29', '68,38', '68,6', '68,70', '68,71', '68,9', '7,11', '7,2', '7,46', '7,47', '7,52', '70,15', '70,43', '70,53', '70,60', '70,7', '73,100', '73,12', '73,2', '73,22', '73,42', '73,47', '73,53', '73,59', '73,70', '73,81', '74,10', '74,15', '74,16', '74,39', '74,75', '74,79', '74,92', '74,94', '74,95', '76,27', '76,33', '76,37', '76,47', '76,48', '76,53', '76,57', '76,71', '76,74', '76,77', '76,85', '76,92', '79,20', '79,37', '79,47', '79,51', '8,24', '8,31', '8,33', '8,34', '8,54', '8,85', '80,37', '80,43', '80,45', '80,52', '80,55', '80,6', '80,65', '82,', '82,15', '82,2', '82,28', '82,3', '82,36', '82,47', '82,48', '82,64', '82,73', '82,74', '85,33', '85,51', '85,53', '85,74', '85,8', '85,80', '85,93', '85,97', '86,12', '86,23', '86,39', '86,69', '86,70', '86,73', '86,83', '86,84', '86,87', '88,15', '88,18', '88,24', '88,36', '88,47', '88,79', '88,81', '88,96', '91,31', '91,32', '91,34', '91,37', '91,44', '91,46', '91,5', '91,77', '91,8', '91,88', '92,10', '92,13', '92,31', '92,58', '92,65', '92,97', '94,24', '94,33', '94,44', '94,52', '94,57', '94,59', '94,61', '94,63', '94,74', '94,82', '94,84', '97,25', '97,59', '97,66', '97,69', '97,70', '97,8', '97,80', '97,86', '98,41', '98,52', '98,62', '98,64', '98,73'], 'filter_dbest': [2451119.0, 2451483.0], 'time2exclude_from_multiprocessing': 0}

    app_client.run(host, port, "select", query)


if __name__ == "__main__":
    main()
