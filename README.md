# TRTD
Source code for paper: "Discovering Topic Representative Terms for Short Text Clustering (IEEE Access)" 

# How to run
``` python kwg_discovery_clustering.py -d dataset/Tweet_merged_50  --gamma 30 --theta 0.8 --delta 0.1```

# Results on Tweet dataset

```
2021-04-03 22:52:01,697 - kwg_discovery_clustering.py - INFO - open dataset: dataset/Tweet_merged_50
2021-04-03 22:52:06,564 - kwg_discovery_clustering.py - INFO - parameters: gamma:30, delta:0.1,theta:0.8
2021-04-03 22:52:06,564 - kwg_discovery_clustering.py - INFO - contruct word graph
2021-04-03 22:52:07,471 - kwg_discovery_clustering.py - INFO - node length:3946, edge length:12980
2021-04-03 22:52:07,476 - kwg_discovery_clustering.py - INFO - ['iphone', 'new', 'app']
2021-04-03 22:52:07,476 - kwg_discovery_clustering.py - INFO - ['flu', 'swine', 'cases']
...
...
... 

2021-04-03 22:52:32,572 - kwg_discovery_clustering.py - INFO - ------------------------clustering result-----------------------------
2021-04-03 22:52:32,572 - kwg_discovery_clustering.py - INFO - original dataset length:167136,pred dataset length:167136
2021-04-03 22:52:32,577 - kwg_discovery_clustering.py - INFO - number of clusters in dataset: 164
2021-04-03 22:52:32,578 - kwg_discovery_clustering.py - INFO - number of clusters estimated: 200
2021-04-03 22:52:32,746 - kwg_discovery_clustering.py - INFO - Homogeneity: 0.846
2021-04-03 22:52:32,874 - kwg_discovery_clustering.py - INFO - Completeness: 0.775
2021-04-03 22:52:32,992 - kwg_discovery_clustering.py - INFO - V-measure: 0.809
2021-04-03 22:52:33,102 - kwg_discovery_clustering.py - INFO - Adjusted Rand Index: 0.842
2021-04-03 22:52:33,925 - kwg_discovery_clustering.py - INFO - Adjusted Mutual Information: 0.771
2021-04-03 22:52:34,074 - kwg_discovery_clustering.py - INFO - Normalized Mutual Information: 0.810
2021-04-03 22:52:34,145 - kwg_discovery_clustering.py - INFO - Purity Score: 0.932
```

# Please cite 

 ```
 @article{yang2019discovering,
  title={Discovering topic representative terms for short text clustering},
  author={Yang, Shuiqiao and Huang, Guangyan and Cai, Borui},
  journal={IEEE Access},
  volume={7},
  pages={92037--92047},
  year={2019},
  publisher={IEEE}
}
```
