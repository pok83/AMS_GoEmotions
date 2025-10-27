# AMS Language Project

**Title:** Divergence between clinical questionnaires and patients’ everyday language revealed by network analysis of symptom vocabulary  
**Authors:** Ichino, K., Okui, H., Horie, T., et al.  
**Version:** Final (October 2025)  
**Target Journal:** *Scientific Reports*  
**Repository:** https://github.com/pok83/AMS_GoEmotions  

---

## 🧭 Overview

This repository contains the complete dataset, code, and manuscript for the AMS Language Project,  
which investigates linguistic divergence between standardized questionnaires and patients’ natural language in late-onset hypogonadism (LOH).  
By combining large-scale natural language processing (NLP) and network analysis, the project compares the vocabulary of LOH questionnaires—especially the Aging Males’ Symptoms (AMS) scale—with two large corpora:

- **SNS Corpus:** GoEmotions dataset (58,009 Reddit comments)
- **Clinical Corpus:** 177 patient-education documents from 59 U.S. medical institutions

---

## 📂 Repository Structure

├── manuscript_AMS.docx # Final manuscript (Scientific Reports submission)
│
├── SNSCorpus.zip # Social-media corpus data (GoEmotions + Reddit subset)
│ ├── edges_weighted10709.csv
│ ├── nodes_2612.xlsx
│ ├── positions_2612.csv
│ ├── bootstrap_outputs_bundle.zip
│ └── figures/ (network plots, community visuals)
│
├── ClinicalCorpus.zip # Patient-education corpus data
│ ├── nodes_6920.xlsx
│ ├── edges_weighted373748.csv
│ ├── Leiden_Louvain_results.csv
│ ├── bootstrap_outputs_clinical.zip
│ └── figures/ (community diagrams, cluster maps)
│
├── code/
│ ├── 01_tokenization.py
│ ├── 02_network_construction.py
│ ├── 03_centrality_analysis.py
│ ├── 04_bootstrap_validation.py
│ └── 05_visualization.ipynb
│
├── results/
│ ├── figures/
│ │ ├── Figure1_network.png
│ │ ├── Figure2_communities.png
│ │ └── Figure3_clinical.png
│ └── tables/
│ ├── Table1_vocabulary.xlsx
│ ├── Table2_centrality.xlsx
│ ├── Table5_bootstrap.xlsx
│
└── README.md


---

## 🧪 Methods Summary

- **Network construction:**  
  Co-occurrence networks built using Positive Pointwise Mutual Information (PPMI) weighting.  
  - Window size = 2 (SNS corpus)  
  - Window size = 10 (clinical corpus)

- **Community detection:**  
  Louvain + Leiden algorithms (resolution = 1.0, random seed = 42)

- **Bootstrap stability:**  
  Poisson resampling (B = 1000 for SNS corpus; B = 400 for clinical corpus)

- **Statistical validation:**  
  Spearman’s ρ, Kendall’s τ, and Jaccard overlap indices for top-k hub sets (k = 10, 20, 50, 100)

- **Software environment:**  
  Python 3.10, NetworkX v2.8, spaCy v3.7, NLTK v3.8, NumPy, Pandas, SciPy, Matplotlib v3.9,  
  Gephi v0.10 (visualization). Executed on Google Colab Pro.

---

## 📊 Key Results

### Social-Media Corpus (GoEmotions)
- 2,612 nodes, 10,709 edges (density = 0.00314)  
- **Anxiety** emerged as the most central and stable hub.  
- Physical and sexual AMS terms (e.g., *libido*, *muscle*) appeared peripherally.  
- Psychological terms dominated the network core, reflecting greater expressibility and social acceptability.

### Clinical Corpus
- 6,920 nodes, 373,748 edges (density = 0.0156)  
- AMS terms were overrepresented in a single integrated cluster connecting sexual, physical, and psychological domains.  
- High centrality values for *muscle*, *erectile*, *weight*, and *libido*.  
- AMS terms acted as **lexical bridges** linking medical and everyday vocabulary.

### Interpretation
- Patients’ spontaneous language emphasizes psychological and emotional terms.  
- Clinical documents emphasize sexual and physical terms for diagnostic clarity.  
- Questionnaires omit contextual cues, forcing isolated interpretation of polysemous terms.  
- This linguistic divergence highlights the need for **linguistically grounded** patient-reported measures.

---

## 🔍 Figures and Tables

- **Figure 1:** Global co-occurrence network (top 300 nodes)  
- **Figure 2:** Community structure (SNS corpus)  
- **Figure 3:** Clinical corpus network (top 300 nodes, AMS terms highlighted)  
- **Table 1:** AMS vocabulary by domain  
- **Table 2:** Centrality of AMS terms  
- **Table 5:** Bootstrap stability metrics  

All figures and tables are available under `/results/`.

---

## 🔐 Data Availability

All datasets are **publicly available and anonymized**:

- **GoEmotions dataset:** [https://github.com/google-research/goemotions](https://github.com/google-research/goemotions)  
- **Reddit corpus:** Used in accordance with Reddit’s Terms of Service (not redistributed)  
- **Clinical corpus:** Compiled from official patient-education pages of 59 U.S. medical institutions  
- No personally identifiable information was collected; all analyses complied with **GDPR** and **HIPAA**.

---

## 💾 Code Availability

All scripts, configuration files, and intermediate outputs are version-controlled within this repository.  
For reproducibility, fix random seeds (42) and execute in Google Colab Pro or Python 3.10 environment.

---

## 📚 Citation

If you use this repository, please cite:

Ichino K., Okui H., Horie T., et al. (2025).
Divergence between clinical questionnaires and patients’ everyday language revealed by network analysis of symptom vocabulary.
Scientific Reports (under review).


---

## 🧠 Contact

For questions or collaboration inquiries:  
📧 [Insert your preferred email or GitHub handle here]

---

© 2025 AMS Language Project. All rights reserved.
