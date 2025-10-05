# AMS_GoEmotions  
*(Bilingual: English / 日本語併記)*  

---

## 📘 Overview / 概要  

This repository provides all data, scripts, and figures used in the AMS–GoEmotions project.  
The study compares linguistic patterns between standardized LOH (Late-Onset Hypogonadism) questionnaires (AMS terms)  
and large-scale social and clinical texts, to explore how symptom vocabulary differs across contexts.  

本リポジトリは、AMSスケール（男性更年期質問票）に基づく語彙と、  
大規模SNSおよび臨床教育文書における自然言語表現を比較し、  
その共起ネットワーク構造を解析した研究の再現データ一式を提供します。  

---

## 📂 Repository Structure / フォルダ構成  

AMS_GoEmotions/
├─ 📄 AMS-GoEmotions.TXT.txt # GoEmotions-based corpus
├─ 📄 AMS_terms.txt # AMS vocabulary list (14 terms)
├─ 📄 AMS_terms_domains.xlsx # AMS terms by domain (psychological / sexual / vitality)
├─ 📄 extra_stop.txt # Additional stopword list
├─ 📄 network_analysis_script.py # Network construction & analysis script
│
├─ 📊 Nodes2612.xlsx # Node attributes (N=2,612)
├─ 📊 edges_weighted10709.csv # Weighted edge list (E=10,709)
│
├─ 📈 cooccurrence_5.csv ... cooccurrence_15.csv # Sensitivity analysis (window size 5–15)
├─ 📈 cooccurrence_all.csv # Merged co-occurrence matrix
├─ 📈 centrality_summary.csv # Summary of centrality metrics
├─ 📈 centrality_table.csv # Raw centrality values
│
├─ 📑 clusters_louvain_summary.csv # Louvain community summary
├─ 📑 clusters_leiden_summary.csv # Leiden community summary
├─ 📑 cluster_level_AMS_vs_nonAMS.csv # AMS vs non-AMS cluster-level stats
├─ 📑 cluster_within_centrality_summary_weighted.csv # Within-cluster centrality
├─ 📑 nodes_with_clusters_and_ranks.csv # Cluster assignment with ranks
├─ 📑 N2612_Louvain_vs_Leiden.xlsx # Louvain vs Leiden comparison
├─ 📑 louvain_leiden_agreement.csv # Agreement indices (NMI, ARI)
│
├─ 📊 AMS_bootstrap_degree_results.xlsx # Bootstrap stability (degree)
├─ 📊 AMS_bootstrap_eigenvector_results.xlsx # Bootstrap stability (eigenvector)
├─ 📊 AMS_bootstrap_betweenness_results.xlsx # Bootstrap stability (betweenness)
│
├─ 🧠 cluster_analysis.xlsx # Cluster-level analysis summary
├─ 🧠 cluster_spread_stats.csv # Cluster size & dispersion metrics
│
├─ 🩺 clinical_corpus/
│ ├─ Mayo Clinic – Male hypogonadism.txt
│ ├─ Stanford Health Care – Low testosterone.txt
│ ├─ UCSF Health – Hypogonadism.txt
│ ├─ UCSF Health – Erectile dysfunction.txt
│ ├─ mayo_hypogonadism_tokens.txt
│ ├─ stanford_lowtestosterone_tokens.txt
│ ├─ ucsf_hypogonadism_tokens.txt
│ ├─ ucsf_ed_tokens.txt
│ └─ tokens_all.txt
│
├─ 📊 AMC.xlsx # Academic medical corpus summary
├─ 📊 LOH_Questionnaires.xlsx # Original AMS/LOH questionnaire list
│
├─ 🎨 Figure1.png # Global co-occurrence network (GoEmotions)
├─ 🎨 Figure2.png # Domain-colored AMS clusters
├─ 🎨 Figure3.png # Clinical corpus network
│
└─ 📁 outputs/ (user-generated)
├─ centrality_all.csv
├─ graph_weighted.gexf
├─ Leiden_summary.xlsx
└─ bootstrap_results/

---

## 🧠 Research Objective / 研究目的  

To quantify lexical and structural differences between standardized questionnaire language (AMS terms)  
and real-world discourse in social and clinical contexts using co-occurrence networks and community detection.  

標準化質問票（AMS）語彙と自然発話言語との語彙的・構造的差異を、  
共起ネットワークおよびクラスタ解析によって定量化します。

---

## ⚙️ Method Summary / 手法概要  

| Step | Process | Key Files |
|------|----------|------------|
| **1. Preprocessing** | Tokenization, lemmatization, stopword removal | `AMS-GoEmotions.TXT.txt`, `extra_stop.txt` |
| **2. Co-occurrence Construction** | Sliding window (size=2–15), PMI normalization | `cooccurrence_*.csv` |
| **3. Network Analysis** | Weighted graph creation, centrality computation | `edges_weighted10709.csv`, `centrality_summary.csv` |
| **4. Community Detection** | Louvain & Leiden algorithms, resolution=1.0/1.5 | `clusters_louvain_summary.csv`, `clusters_leiden_summary.csv` |
| **5. Bootstrap Stability** | 1,000 resamples for AMS term stability | `AMS_bootstrap_*.xlsx` |
| **6. Visualization** | Top 300 nodes with AMS labels and domain colors | `Figure1.png`, `Figure2.png`, `Figure3.png` |

---

## ▶️ Reproduction / 解析再現手順　

### 1️⃣ Install environment　

```bash
pip install pandas numpy nltk networkx matplotlib python-louvain igraph leidenalg

### 2️⃣ Run main analysis　

python network_analysis_script.py \
  --input "AMS-GoEmotions.TXT.txt" \
  --ams_terms "AMS_terms.txt" \
  --stopwords "extra_stop.txt" \
  --window 2 \
  --outdir "outputs"

### 3️⃣ Optional: Sensitivity analysis
python network_analysis_script.py --window 5
python network_analysis_script.py --window 7
python network_analysis_script.py --window 15

4️⃣ Visualization
Use any network viewer (e.g., Gephi, Cytoscape) to open:
network_top300_withpos.graphml
graph_weighted.gexf


## 📊 Key Results / 主な結果
| Figure       | Description                                                                             |
| ------------ | --------------------------------------------------------------------------------------- |
| **Figure 1** | Global co-occurrence network (AMS terms in red, non-AMS in blue).                       |
| **Figure 2** | Domain-colored clusters (psychological=purple, sexual=orange, vitality/physical=green). |
| **Figure 3** | Clinical corpus network showing AMS-related lexical clusters in educational texts.      |

## 🧩 Related Data Summaries / 補足データ
| File                                 | Content                                                 |
| ------------------------------------ | ------------------------------------------------------- |
| `centrality_summary.csv`             | Mean/SD of degree, eigenvector, betweenness             |
| `cluster_level_AMS_vs_nonAMS.csv`    | Comparison of AMS vs non-AMS terms within clusters      |
| `louvain_leiden_agreement.csv`       | Agreement indices (NMI, ARI) between clustering methods |
| `AMS_terms_within_cluster_ranks.csv` | Rank of AMS terms within their communities              |
| `cluster_spread_stats.csv`           | Cluster size, edge density, domain dispersion           |

## 🩺 Clinical Corpus Overview / 臨床コーパス概要
| Source                   | Document                              | Focus                                           |
| ------------------------ | ------------------------------------- | ----------------------------------------------- |
| **Mayo Clinic**          | *Male hypogonadism*                   | Symptoms, causes, treatment overview            |
| **Stanford Health Care** | *Low testosterone*                    | Clinical overview and endocrine/urology context |
| **UCSF Health**          | *Hypogonadism / Erectile dysfunction* | Endocrine and urologic patient education texts  |

## 🧮 Environment / 実行環境
Python 3.10
NetworkX v2.8
igraph + leidenalg
pandas, numpy, scikit-learn
matplotlib, unidecode, nltk
Random seed fixed at 42 for reproducibility.

👤 Contact / 連絡先

Author: Kenta Ichino
Affiliation: Juntendo University, Department of Urology
Email: k.ichino.xc@juntendo.ac.jp
---
