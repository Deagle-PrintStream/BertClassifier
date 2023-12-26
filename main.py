# coding: utf-8
# @File: main.py
# @Author: Swayee
# @Email: suiyiliu@mail.ustc.edu.cn
# @Time: 2023/12/26

import os,sys
import warnings
import argparse
import logging
import datetime
from train import train
from predict import predict

def parse_argument() -> argparse.Namespace:
    """read in arguments from command line, kernel arguments:
    configuration file, random seed, test missing ratio, folder number in CV,
    pretrained model to load(or not), count of samples.
    """
    parser = argparse.ArgumentParser(description="CSDI")
    parser.add_argument("--config", type=str, default="./config/test.yaml")

    args = parser.parse_args()
    print(args)
    logging.info(args)
    return args

def main() -> None:

    os.chdir(sys.path[0])
    warnings.filterwarnings("ignore")

    #read arguments, yaml config files and save it
    args = parse_argument()
    #loggin setting
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.DEBUG,
        filename="./save/log" + current_time + ".log",
        format="%(asctime)s %(message)s",
        datefmt="%I:%M:%S ",
    )

    pass

if __name__ == '__main__':
    main()

