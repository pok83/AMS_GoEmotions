## 00_network_summaries_copy.csv
- 形状: 3行 × 8列
- 列: w, N_tokens, V, E, Q_modularity, density, avg_degree, avg_clustering
- Nodes (N): **6920**
- Edges (E): **223520**
- Density: **0.009337**
- Average degree: **64.6**

## 01_ams_vs_nonams.csv
- 形状: 4行 × 7列
- 列: metric, median_AMS, median_nonAMS, MWU_stat, p_value, cliffs_delta, p_adj_fdr
- 読み込み/解釈エラー: None

## 02_comm_overrep.csv
- 形状: 12行 × 6列
- 列: community, N_c, K_c_AMS, prop_AMS_c, p_overrep, p_adj_fdr

## 03_top20_deg.csv
- 形状: 20行 × 3列
- 列: node, is_AMS, deg
- AMS含有数: **0 / 20**
- 上位リスト（先頭10）: testosterone, man, level, not, low, health, therapy, treatment, patient, hormone ...

## 03_top20_eigenvector.csv
- 形状: 20行 × 3列
- 列: node, is_AMS, eigenvector
- AMS含有数: **0 / 20**
- 上位リスト（先頭10）: patient, not, time, health, treatment, like, testosterone, therapy, man, include ...

## 03_top20_pagerank.csv
- 形状: 20行 × 3列
- 列: node, is_AMS, pagerank
- AMS含有数: **0 / 20**
- 上位リスト（先頭10）: testosterone, man, health, not, therapy, patient, like, treatment, level, study ...

## AMS_centrality_w10_only.csv
- 形状: 13行 × 14列
- 列: node, community, deg, pagerank, eigenvector, betweenness, rank_deg, rank_pagerank, rank_eigenvector, rank_betweenness, pct_deg, pct_pagerank, pct_eigenvector, pct_betweenness
- Top degree: muscle (1015)
- Top eigenvector: weight (0.03363)
- Top PageRank: erectile (0.0006756)
- 語数: 13

## AMS_community_distribution_w10.csv
- 形状: 2行 × 3列
- 列: community, n_AMS_terms, prop_AMS_terms
- AMS含有クラスタ数: **2**
- クラスタ別AMS語数（上位）: 0=1, 6=1

## AMS_results_w10_significant.xlsx
- 形状: 4行 × 8列
- 列: metric, median_AMS, median_nonAMS, MWU_stat, p_value, cliffs_delta, p_adj_fdr, sig
- p<0.05 行: **4 / 4**

## AMS_ego_neighbors_w10.csv
- 形状: 260行 × 4列
- 列: seed, neighbor, weight, ams_neighbor
- 読み込み/解釈エラー: You have to supply one of 'by' and 'level'

## fig_ams_ego_w10_k2_blueEdges_positions.csv
- 形状: 140行 × 4列
- 列: node, x, y, is_AMS
- Ego図の座標データ（行数＝ノード数相当）。用途：Egoネットワーク図の座標割当。

## cluster_results_louvain.xlsx
- 形状: 6920行 × 10列
- 列: node, deg, deg_norm, betweenness, eigenvector, pagerank, community, node_norm, is_ams, cluster
- 形式不明（単一シートと仮定したが解析対象列が不足）

## Leiden_vs_Louvain.xlsx
- 形状: 4行 × 8列
- 列: resolution, Q_leiden, Q_louvain, NMI, ARI, n_nodes, n_clusters_leiden, n_clusters_louvain
- 形式不明

## perm_overrep_louvain.csv
- 形状: 13行 × 2列
- 列: cluster, perm_p

## cluster1_ams_subclusters.csv
- 形状: 12行 × 7列
- 列: node, domain, degree, w_degree, betweenness, eigenvector, subcluster
- サブクラスタ数: **3**
- サブクラスタ別AMS語数: 0=5, 2=4, 1=3

## subgraph_w10_nodes.csv
- 形状: 313行 × 5列
- 列: node, is_AMS, size, degree_metric, domain
- ノード表。ユニークノード数: **313**

## subgraph_w10_edges.csv
- 形状: 30402行 × 3列
- 列: source, target, weight
- エッジ表。行数=エッジ数: **30402**
- 重み（weight）: min=1.832e-06, median=1.204, max=8.2

## subgraph_w10_positions.csv
- 形状: 313行 × 4列
- 列: node, x, y, is_AMS
- 座標表。ノード座標の行数: **313**
- AMS語カバレッジ: **13 / 14**（不足: sweating）

## edges_ppmi_w10.csv
- 形状: 373748行 × 3列
- 列: u, v, ppmi
- PPMI分布: min=1.832e-06, q25=2.029, median=3.637, q75=5.802, max=13.47

## fig_network_w10_positions.csv
- 形状: 6594行 × 4列
- 列: node, x, y, is_AMS
- 座標表。ノード座標の行数: **6594**
- AMS語カバレッジ: **6 / 14**（不足: erectile, erection, gain, hair, libido, muscle, sweating, weight）

## network_summaries.csv
- 形状: 3行 × 8列
- 列: w, N_tokens, V, E, Q_modularity, density, avg_degree, avg_clustering
- Nodes (N): **6920**
- Edges (E): **223520**
- Density: **0.009337**
- Average degree: **64.6**

## nodes_centrality_w10.csv
- 形状: 6920行 × 7列
- 列: node, deg, deg_norm, betweenness, eigenvector, pagerank, cluster
- degree 上位: testosterone, man, level, not, low, health, therapy, treatment, patient, hormone
- eigenvector 上位: patient, not, time, health, treatment, like, testosterone, therapy, man, include
- PageRank 上位: testosterone, man, health, not, therapy, patient, like, treatment, level, study

## AMS_terms_domains.xlsx
- 形状: 7行 × 3列
- 列: Psychological, Sexual, Vitality/Physical
- AMS語セル数: **14**
