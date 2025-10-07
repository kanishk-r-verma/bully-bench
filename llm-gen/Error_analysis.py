"""
Conversation-level similarity analysis + metadata overlays
==========================================================

Inputs
------
  • llm_both_trainEval.csv
  • llm_both_test.csv

Outputs (saved in ./results/)
-----------------------------
  ├── inter_scenario_similarity_matrix.png         # heat-map with diagonal
  ├── inter_scenario_similarity.csv                # numeric matrix
  ├── cross_scenario_similar_pairs.csv             # top cross-scenario pairs
  ├── cross_scenario_network.png                   # spring-layout graph
  ├── subtype_similarity_barplot.png               # avg sim by aggression type
  └── summary_report.md                            # quick prose recap
"""

# --------------------------------------------------------------------------
# 0 ─ Imports & configuration
# --------------------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import torch
import time, warnings, json

warnings.filterwarnings("ignore")
sns.set_palette("viridis")
plt.style.use("seaborn-v0_8-whitegrid")

CONFIG = {
    "train_eval": "llm_both_trainEval.csv",
    "test":        "llm_both_test.csv",
    "file_pattern": "_Lit_",
    "model_name":  "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size":  64,
    "sim_thresh":  0.50,
    "out":         Path("results"),
}

# --------------------------------------------------------------------------
# 1 ─ Static metadata (extend here if more scenarios appear later)
# --------------------------------------------------------------------------
meta_records = [
    ("Lit_1",  "Threats",       "Offline→Online", "Male"),
    ("Lit_10", "Exclusion",     "Offline→Online", "Female"),
    ("Lit_11", "Exclusion",     "Offline→Online", "Male"),
    ("Lit_12", "Flaming",       "Online",         "Female"),
    ("Lit_13", "Flaming",       "Online",         "Female"),
    ("Lit_14", "Harassment",    "Offline→Online", "Female"),
    ("Lit_15", "Impersonation", "Online",         "Female"),
    ("Lit_2",  "Threats",       "Offline→Online", "Female"),
    ("Lit_3",  "Outing",        "Offline→Online", "Female"),
    ("Lit_4",  "Outing",        "Offline→Online", "Female"),
    ("Lit_5",  "Outing",        "Offline→Online", "Female"),
    ("Lit_6",  "Denigration",   "Online",         "Female"),
    ("Lit_7",  "Denigration",   "Online",         "Female"),
    ("Lit_8",  "Denigration",   "Online",         "Male"),
    ("Lit_9",  "Exclusion",     "Online",         "Female"),
]
META = pd.DataFrame(meta_records, columns=["scenario", "subtype", "mode", "target_gender"])

# --------------------------------------------------------------------------
# 2 ─ Helper functions
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = (
        pd.concat([pd.read_csv(CONFIG["train_eval"]), pd.read_csv(CONFIG["test"])])
        .query("FILE.str.contains(@CONFIG['file_pattern'])", engine="python")
        .assign(
            run=lambda d: d.FILE.str.split("_").str[0],
            scenario=lambda d: d.FILE.str.split("_").str[1:].str.join("_"),
        )
        .dropna(subset=["TEXT", "ROLE"])
    )
    df = df[df.TEXT.str.strip() != ""]
    return df

def group_conversations(df: pd.DataFrame):
    conv = {}
    for (sc, run), block in df.groupby(["scenario", "run"]):
        text = " | ".join(f"{r['ROLE']}: {r['TEXT']}" for _, r in block.iterrows())
        if text:
            key = f"{sc}_{run}"
            conv[key] = {
                "scenario": sc,
                "run": run,
                "text": text,
            }
    return conv

def encode(conv, model):
    texts = [c["text"] for c in conv.values()]
    return model.encode(texts, batch_size=CONFIG["batch_size"],
                        convert_to_tensor=True, show_progress_bar=True)

def save_heatmap(mat, labels, path, title):
    plt.figure(figsize=(0.8*len(labels)+4, 0.7*len(labels)+3))
    sns.heatmap(mat, xticklabels=labels, yticklabels=labels,
                cmap="cividis", annot=True, fmt=".2f")
    plt.title(title, pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# --------------------------------------------------------------------------
# 3 ─ Main
# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    CONFIG["out"].mkdir(exist_ok=True)

    # 3.1  Data prep
    df = load_data()
    conv = group_conversations(df)
    keys = list(conv.keys())
    print(f"✅ {len(keys)} conversations across {df.scenario.nunique()} scenarios loaded.")

    # 3.2  Embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(CONFIG["model_name"], device=device)
    emb = encode(conv, model).cpu()
    sim = util.cos_sim(emb, emb).numpy()

    # 3.3  Inter-scenario similarity matrix
    scen_index = {sc: i for i, sc in enumerate(df.scenario.unique())}
    scen_names = list(scen_index.keys())
    inter = np.zeros((len(scen_names), len(scen_names)))
    run_lookup = defaultdict(list)
    for i, k in enumerate(keys):
        run_lookup[conv[k]["scenario"]].append(i)
    for i, s1 in enumerate(scen_names):
        for j, s2 in enumerate(scen_names):
            idx1, idx2 = run_lookup[s1], run_lookup[s2]
            block = sim[np.ix_(idx1, idx2)]
            if i == j and len(idx1) > 1:  # intra-scenario, drop diagonal
                m = block[np.triu_indices_from(block, k=1)].mean()
            else:
                m = block.mean()
            inter[i, j] = m

    # 3.4  Persist matrix
    pd.DataFrame(inter, index=scen_names, columns=scen_names)\
      .to_csv(CONFIG["out"]/ "inter_scenario_similarity.csv", float_format="%.4f")
    save_heatmap(
        inter, scen_names,
        CONFIG["out"]/ "inter_scenario_similarity_matrix.png",
        "Inter-Scenario Average Similarity Matrix\n(Diagonal = intra-scenario avg.)",
    )

    # 3.5  Cross-scenario pairs csv + network
    pairs = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            sc1, sc2 = conv[keys[i]]["scenario"], conv[keys[j]]["scenario"]
            if sc1 != sc2:
                pairs.append({
                    "conv1": keys[i], "conv2": keys[j],
                    "scenario1": sc1, "scenario2": sc2,
                    "similarity": sim[i, j],
                })
    cross_df = pd.DataFrame(pairs).sort_values("similarity", ascending=False)
    cross_df.to_csv(CONFIG["out"]/ "cross_scenario_similar_pairs.csv", index=False)

    # network graph (edges above threshold)
    G = nx.Graph()
    for _, r in cross_df.query("similarity > @CONFIG['sim_thresh']").iterrows():
        G.add_edge(r.scenario1, r.scenario2, weight=r.similarity)
    if G.number_of_edges():
        plt.figure(figsize=(14, 14))
        pos = nx.spring_layout(G, k=0.6, iterations=100, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=900, node_color="skyblue")
        edges = nx.draw_networkx_edges(
            G, pos,
            width=[G[u][v]["weight"]*8 for u, v in G.edges()],
            edge_color=[G[u][v]["weight"] for u, v in G.edges()],
            edge_cmap=plt.cm.magma)
        nx.draw_networkx_labels(G, pos, font_weight="bold")
        plt.title("Cross-Scenario Similarity Network (≥0.50)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(CONFIG["out"]/ "cross_scenario_network.png", dpi=300)
        plt.close()

    # 3.6  Metadata overlay – average similarity by subtype
    scen_meta = META.set_index("scenario")
    intra_means = []
    for sc, idx in run_lookup.items():
        if len(idx) < 2:
            continue
        m = sim[np.ix_(idx, idx)]
        intra = m[np.triu_indices_from(m, k=1)].mean()
        intra_means.append({"scenario": sc, "intra_mean": intra})
    intra_df = pd.DataFrame(intra_means).merge(scen_meta, left_on="scenario", right_index=True)
    bar = (intra_df.groupby("subtype")["intra_mean"].mean()
           .sort_values(ascending=False).reset_index())
    plt.figure(figsize=(10,6))
    sns.barplot(data=bar, x="subtype", y="intra_mean")
    plt.xlabel("Cyber-aggression subtype")
    plt.ylabel("Avg. intra-scenario similarity")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title("Average intra-scenario similarity by subtype")
    plt.tight_layout()
    plt.savefig(CONFIG["out"]/ "subtype_similarity_barplot.png", dpi=300)
    plt.close()

    # 3.7  Prose summary
    summary = f"""# Similarity Analysis Summary
*Total conversations:* {len(keys)}
*Distinct scenarios:* {df.scenario.nunique()}

**Intra-scenario similarity**  
Mean: {intra_df.intra_mean.mean():.3f} ± {intra_df.intra_mean.std():.3f}

**Highest cross-scenario similarity pair**  
{cross_df.iloc[0].scenario1} ↔ {cross_df.iloc[0].scenario2}  (sim={cross_df.iloc[0].similarity:.2f})

See `inter_scenario_similarity_matrix.png` for the full matrix,
`cross_scenario_network.png` for a graph view, and
`subtype_similarity_barplot.png` for subtype trends.
"""
    (CONFIG["out"]/ "summary_report.md").write_text(summary)

    # --------------------------------------------------------------------------
    # 3.8  Group-level similarity analytics  (category / mode / gender)
    # --------------------------------------------------------------------------
    def group_similarity(sim_mat: np.ndarray,
                         conv_keys: list[str],
                         conv_lookup: dict,
                         meta: pd.DataFrame,
                         field: str,
                         out_prefix: str):
        """
        Compute within-group stats and a group-by-group similarity matrix.
        Saves two CSVs and one heat-map.
        """
        #
        # 1.  Map every conversation -> group label
        #
        label_of = {k: meta.loc[conv_lookup[k]['scenario'], field]
                    for k in conv_keys}
        
        groups = defaultdict(list)
        for idx, k in enumerate(conv_keys):
            groups[label_of[k]].append(idx)
        labels = sorted(groups)
        
        #
        # 2.  Within-group statistics
        #
        stats_rows = []
        for lab, idxs in groups.items():
            if len(idxs) < 2:
                continue
            block = sim_mat[np.ix_(idxs, idxs)]
            tri = block[np.triu_indices_from(block, k=1)]
            stats_rows.append({
                field: lab,
                "count": len(idxs),
                "mean":  tri.mean(),
                "std":   tri.std(),
                "min":   tri.min(),
                "max":   tri.max()
            })
        stats_df = pd.DataFrame(stats_rows).sort_values("mean", ascending=False)
        stats_df.to_csv(CONFIG["out"]/f"{out_prefix}_within_stats.csv", index=False)
        
        #
        # 3.  Group-vs-group matrix
        #
        m = np.zeros((len(labels), len(labels)))
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                block = sim_mat[np.ix_(groups[a], groups[b])]
                if i == j:
                    if len(groups[a]) > 1:
                        tri = block[np.triu_indices_from(block, k=1)]
                        m[i, j] = tri.mean()
                    else:
                        m[i, j] = np.nan        # only one item in group
                else:
                    m[i, j] = block.mean()
        mat_df = pd.DataFrame(m, index=labels, columns=labels)
        mat_df.to_csv(CONFIG["out"]/f"{out_prefix}_matrix.csv", float_format="%.3f")
        
        #
        # 4.  Heat-map
        #
        save_heatmap(
            m, labels,
            CONFIG["out"]/f"{out_prefix}_matrix.png",
            f"{field.title()} ↔ {field.title()} similarity matrix"
        )
        print(f"   · {field}: stats+matrix written.")

    # --------------------------------------------------------------------------
    # 3.9  Run group analyses
    # --------------------------------------------------------------------------
    #  • look-ups for scenario → metadata (meta is already indexed by scenario)
    META.set_index("scenario", inplace=True)

    group_similarity(sim, keys, conv, META, "subtype",        "subtype")
    group_similarity(sim, keys, conv, META, "mode",           "mode")
    group_similarity(sim, keys, conv, META, "target_gender",  "gender")

    print(f"✅ Done. All artefacts saved to {CONFIG['out']}  "
          f"({time.time()-t0:.1f}s elapsed).")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
