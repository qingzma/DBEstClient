# Created by Qingzhi Ma at 2019-07-25
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk
from dbestclient.cli.prompt import DBEstPrompt
from dbestclient.ml import regtf,regtf2,mdn,regtorch
import argparse

def main():
    p = DBEstPrompt()
    p.cmdloop()

def cmd():
    print("Welcome to DBEst++")

    parser = argparse.ArgumentParser(description='Process the input for DBEst++.')
    # parser.add_argument('--foo', action='store')
    parser.add_argument('--pm25', action='store_true', help="run pm25 experiments")
    args = parser.parse_args()

    if args.pm25:
        print("run pm25")
        regtorch.tutorial()

if __name__=="__main__":
    main()
