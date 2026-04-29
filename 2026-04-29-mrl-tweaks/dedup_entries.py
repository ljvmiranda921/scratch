import argparse

import pandas as pd


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path")
    parser.add_argument("-o", "--output_path")
    # fmt: on
    return parser.parse_args()


def main():
    pass


if __name__ == "__main__":
    main()
