#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMS-GoEmotions Network Analysis (Colab-ready)
---------------------------------------------
Build a word co-occurrence network from AMS-GoEmotions.* (TXT/CSV/TSV/JSONL),
block function words (I, was, the, etc.) so they can't become hubs, and check
whether AMS-important terms act as hubs.

Example (in Colab):
!pip -q install pandas numpy nltk networkx matplotlib unidecode scikit-learn python-louvain
!python ams_goemotions_network.py \
  --input "AMS-GoEmotions.TXT" \
  --language en \
  --window 2 \
  --min_count 3 \
  --max_vocab 20000 \
  --ams_terms "AMS_terms.txt" \
  --outdir "outputs" \
  --top_plot 150

Outputs (in outdir):
- centrality_all.csv
- centrality_ams_only.csv
- edges_weighted.csv
- vocab.csv
- graph_weighted.gexf
- network_top.png
- params.json
"""
import argparse, json, math, re, sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from unidecode import unidecode

try:
    import community as community_louvain  # python-louvain
except Exception:
    community_louvain = None

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer


def read_table_auto(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in ['.jsonl', '.json']:
        rows = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rows.append(obj)
                except Exception:
                    rows.append({'text': line})
        return pd.DataFrame(rows if rows else [{'text': ''}])
    # try common separators
    for sep in [',', '\t', '|', ';']:
        try:
            df = pd.read_csv(path, sep=sep, engine='python')
            if df.shape[1] >= 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path, engine='python')


def guess_text_column(df: pd.DataFrame) -> str:
    for name in ['text', 'body', 'comment', 'sentence', 'content', 'utterance', 'message']:
        if name in df.columns:
            return name
    object_cols = [c for c in df.columns if df[c].dtype == object]
    candidates = object_cols if object_cols else df.columns.tolist()
    lens = df[candidates].applymap(lambda x: len(str(x))).mean().sort_values(ascending=False)
    return lens.index[0]


def build_stopword_set(language: str = 'en') -> set:
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    sw = set(stopwords.words('english')) if language.startswith('en') else set()
    extra = {
        "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
        "yourself","yourselves","he","him","his","himself","she","her","hers","herself",
        "it","its","itself","they","them","their","theirs","themselves",
        "what","which","who","whom","this","that","these","those",
        "am","is","are","was","were","be","been","being",
        "have","has","had","having","do","does","did","doing",
        "a","an","the","and","but","if","or","because","as","until","while",
        "of","at","by","for","with","about","against","between","into","through",
        "during","before","after","above","below","to","from","up","down","in","out",
        "on","off","over","under","again","further","then","once","here","there",
        "when","where","why","how","all","any","both","each","few","more","most",
        "other","some","such","no","nor","not","only","own","same","so","than","too",
        "very","s","t","can","will","just","don","should","now"
    }
    artifacts = {"—","–","…","’","‘","“","”","'","\"","``","''","..",".",",",";","?","!","(",")","[","]","{","}","/","\\","|","&","*","#","%","$","@",":","-","_","’re","’s","n’t"}
    return sw | extra | artifacts


def tokenize(text: str, language: str = 'en') -> List[str]:
    t = unidecode(str(text)).lower()
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'@[A-Za-z0-9_]+', ' ', t)
    t = re.sub(r'[^a-z0-9\s\-]+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    toks = []
    for tok in t.split():
        toks.extend(tok.split('-'))
    if language.startswith('en'):
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        toks = [lemmatizer.lemmatize(w) for w in toks]
    return [w for w in toks if w]


def cooccurrence_edges(tokens: List[str], window: int = 2) -> Counter:
    edges = Counter()
    L = len(tokens)
    for i, w in enumerate(tokens):
        left = max(0, i - window)
        right = min(L, i + window + 1)
        for j in range(left, right):
            if j <= i:
                continue
            u, v = w, tokens[j]
            if u == v:
                continue
            if u > v:
                u, v = v, u
            edges[(u, v)] += 1
    return edges


def pmi_weights(edges: Dict[Tuple[str, str], int], counts: Dict[str, int]) -> Dict[Tuple[str, str], float]:
    total = sum(counts.values())
    pw = {}
    for (u, v), c in edges.items():
        p_u = counts[u] / total
        p_v = counts[v] / total
        p_uv = c / total
        denom = p_u * p_v if p_u * p_v > 0 else 1e-12
        pmi = math.log2(p_uv / denom)
        pw[(u, v)] = max(0.0, pmi)  # positive PMI
    return pw


def infer_ams_terms(texts: List[str], top_k: int, stop: set) -> List[str]:
    vec = TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b[A-Za-z0-9][A-Za-z0-9\-]+\b', max_features=20000)
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    max_scores = X.max(axis=0).toarray().ravel()
    order = np.argsort(max_scores)[::-1]
    return [t for t in terms[order] if t not in stop][:top_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="AMS-GoEmotions.* (TXT/CSV/TSV/JSONL)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--min_count", type=int, default=3, help="drop tokens below this freq")
    ap.add_argument("--max_vocab", type=int, default=20000)
    ap.add_argument("--ams_terms", default="", help="optional file (one term per line)")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--top_plot", type=int, default=150, help="plot top-N nodes by degree")
    args = ap.parse_args()

    path = Path(args.input)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stop = build_stopword_set(args.language)

    # Load table
    df = read_table_auto(path)
    text_col = guess_text_column(df)
    texts = df[text_col].astype(str).tolist()

    # Tokenize & count
    tokens_all = []
    for t in texts:
        toks = [w for w in tokenize(t, args.language) if w not in stop]
        tokens_all.extend(toks)
    # limit vocab by freq
    counts = Counter(tokens_all)
    vocab = [w for w, c in counts.most_common(args.max_vocab) if c >= args.min_count]
    vocab = set(vocab)
    tokens_all = [w for w in tokens_all if w in vocab]
    counts = Counter(tokens_all)

    # Build edge counts (single concatenated stream; simple & effective)
    edges = cooccurrence_edges(tokens_all, window=args.window)
    ppmi = pmi_weights(edges, counts)

    # Build graph
    G = nx.Graph()
    for (u, v), c in edges.items():
        w = ppmi[(u, v)]
        if w > 0.0:
            G.add_edge(u, v, weight=float(w), count=int(c))

    # Centralities
    if len(G) == 0:
        print("Graph is empty after filtering. Check parameters or input.")
        sys.exit(0)
    deg = nx.degree_centrality(G)
    bet = nx.betweenness_centrality(G, normalized=True)
    try:
        eig = nx.eigenvector_centrality_numpy(G)
    except Exception:
        eig = {n: 0.0 for n in G.nodes()}

    cent_df = pd.DataFrame({
        'node': list(G.nodes()),
        'degree': [deg[n] for n in G.nodes()],
        'betweenness': [bet[n] for n in G.nodes()],
        'eigenvector': [eig[n] for n in G.nodes()],
        'freq': [counts[n] for n in G.nodes()],
    }).sort_values('degree', ascending=False)
    cent_df.to_csv(outdir / "centrality_all.csv", index=False)

    # AMS terms
    ams_terms = []
    if args.ams_terms and Path(args.ams_terms).exists():
        with open(args.ams_terms, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                term = line.strip().lower()
                if term and term not in stop:
                    ams_terms.append(term)
    else:
        # fallback: infer from TF-IDF
        ams_terms = infer_ams_terms(texts, top_k=100, stop=stop)

    ams_set = set(ams_terms)
    cent_ams = cent_df[cent_df['node'].isin(ams_set)].copy()
    cent_ams.sort_values('degree', ascending=False, inplace=True)
    cent_ams.to_csv(outdir / "centrality_ams_only.csv", index=False)

    # Export edges & vocab
    edges_rows = [{'u': u, 'v': v, 'count': c, 'ppmi': ppmi[(u, v)]} for (u, v), c in edges.items() if ppmi[(u, v)] > 0.0]
    pd.DataFrame(edges_rows).to_csv(outdir / "edges_weighted.csv", index=False)
    pd.DataFrame({'token':[w for w,_ in counts.most_common()], 'count':[c for _,c in counts.most_common()]}).to_csv(outdir / "vocab.csv", index=False)

    # Communities (optional)
    if community_louvain is not None and len(G) > 0:
        part = community_louvain.best_partition(G, weight='weight', resolution=1.0)
        nx.set_node_attributes(G, part, 'community')

    # Save GEXF
    nx.write_gexf(G, outdir / "graph_weighted.gexf")

    # Quick plot (top-N by degree)
    topN = cent_df.head(args.top_plot)['node'].tolist()
    H = G.subgraph(topN).copy()
    pos = nx.spring_layout(H, seed=42, k=None, weight='weight')
    sizes = [300 + 3000*deg[n] for n in H.nodes()]
    plt.figure(figsize=(12, 9), dpi=180)
    nx.draw_networkx_nodes(H, pos, node_size=sizes)
    nx.draw_networkx_edges(H, pos, width=0.5)
    nx.draw_networkx_labels(H, pos, font_size=8)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outdir / "network_top.png")
    plt.close()

    # Params
    params = dict(
        input=str(path.name),
        language=args.language,
        window=args.window,
        min_count=args.min_count,
        max_vocab=args.max_vocab,
        ams_terms_file=args.ams_terms if args.ams_terms else "(inferred TF-IDF)",
        outdir=str(outdir),
        nodes=len(G.nodes()),
        edges=len(G.edges())
    )
    with open(outdir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print("Done. Outputs written to:", outdir)


if __name__ == "__main__":
    main()
