





import argparse
from data_util import convert_raw_data
from Data import Data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', type=str, default='small', help='')

    args = parser.parse_args()
    print(args)


    #convert_raw_data(args.dataset)

    data = Data.load(args.dataset)
    data.print_statisitcs()
    data.split_setting = [20, 500]
    data.update_mask()

    # accessing adjacecny matrix
    A = data.raw_adj.to_dense()  # data.adj is sparse tensor

    # accessing normalied adjacency matrix
    norm_A = data.norm_adj.to_dense()

    exit()