# STCommunityDetection

Datasets and codes for the paper 'GGT: A Spatio-Temporal Gated Graph Transformer with Node-Edge Joint Attention Mechanism for Community Detection' are provided here.

To run GGT, follow the steps below:

1. Download the Gowalla, Brightkite and Weeplaces datasets, and then extract the datasets to the following directory.

   ```text
   ├── STCommunityDetection
   	├── dataset
       	├── brightkite
       		├── Brightkite_edges.txt
       		├── Brightkite_totalCheckins.txt
       	├── gowalla
       		├── Gowalla_edges.txt
       		├── Gowalla_totalCheckins.txt
       	├── weeplaces
   			├── weeplace_checkins.csv
   			├── weeplace_friends.csv
   ```

2. run main.py

```bash
python main.py
```

## Dataset

1. [Gowalla](https://snap.stanford.edu/data/loc-gowalla.html)
2. [Brightkite](https://snap.stanford.edu/data/loc-brightkite.html)
3. [Weeplaces](https://www.yongliu.org/datasets/)

