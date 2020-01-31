# Dataset for node classificatino task

Node classification data set.

## Usage

```
data = Data.load('cora')
data.print_statisitcs()

# data randomly sample training nodes and validation nodes for experiments
# 20 is the num. training nodes for EACH CLASS, 500 is the num. validation nodes among ALL nodes.
data.split_setting = [20, 500] # or you can modify split.txt in /data/{dataset_name} to set this
data.update_mask()

# accessing adjacecny matrix(torch.Tensor)
A = data.raw_adj.to_dense() # data.adj is pytorch sparse tensor

# accessing normalied adjacency matrix(torch.Tensor)
norm_A = data.norm_adj.to_dense()

# features(torch.Tensor)
data.features

# labels(torch.Tensor)
data.labels
```

See the Data.py file for the details.


## NOTE!!
- In the dataset, it ONLY contains the largest component!! If you want to use WHOLE graph(s) please modify 'convert_raw_data' function.
- Do not use this repository for paper works. PLEASE cite appropriate github repos, and proper papers for the tasks. You may refer to the Thanks section below.

## Thanks
- https://github.com/kimiyoung/planetoid
- https://github.com/graphdml-uiuc-jlu/geom-gcn
- https://github.com/marble0117/gcnns

## Lisence
MIT