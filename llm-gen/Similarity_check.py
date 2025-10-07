import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import torch
from sentence_transformers import SentenceTransformer, util
import os
from tqdm import tqdm
import time
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import warnings
from typing import List, Dict, Any, Tuple

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
CONFIG = {
    "train_eval_data": 'llm_both_trainEval.csv',
    "test_data": 'llm_both_test.csv',
    "file_pattern": '_Lit_',
    "output_dir": "conversation_analysis_results",
    "model_name": 'all-MiniLM-L6-v2',
    "batch_size": 64,
    "similarity_threshold": 0.5,
    "high_similarity_threshold": 0.8,
}

# --- Plotting Style Setup ---
def setup_plotting_style():
    """Sets up a professional and consistent plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")
    plt.rcParams.update({
        'figure.figsize': (14, 8),
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 100,
    })
    print("🎨 Plotting style configured.")

# --- Core Helper Functions ---
def get_device() -> str:
    """Detects and returns the appropriate device for Torch (GPU or CPU)."""
    if torch.cuda.is_available():
        print(f"✅ Found {torch.cuda.device_count()} CUDA device(s). Using GPU.")
        return 'cuda'
    print("⚠️ No CUDA devices found. Using CPU. This may be slow.")
    return 'cpu'

def load_and_prepare_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Loads, combines, and preprocesses the conversation data."""
    print("📖 Loading and preparing datasets...")
    try:
        df1 = pd.read_csv(config['train_eval_data'])
        df2 = pd.read_csv(config['test_data'])
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Make sure the CSV files are in the correct directory.")
        return pd.DataFrame()

    total = pd.concat([df1, df2])
    print(f"   Combined dataset has {len(total)} rows.")
    
    # Filter by the specified pattern
    total = total[total.FILE.str.contains(config['file_pattern'], na=False)]
    print(f"   Filtered to {len(total)} rows with pattern '{config['file_pattern']}'.")

    # Extract scenario and run
    total['run'] = total['FILE'].apply(lambda x: x.split('_')[0])
    total['scenario'] = total['FILE'].apply(lambda x: '_'.join(x.split('_')[1:]))
    
    # Clean data
    total = total.dropna(subset=['TEXT', 'ROLE', 'scenario', 'run'])
    total = total[total['TEXT'].str.strip() != '']
    print(f"   Cleaned data has {len(total)} rows.")
    
    return total

def group_conversations(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Groups the dataframe into distinct conversations."""
    print("🔄 Grouping messages into conversations...")
    conversations = {}
    
    def get_conversation_text(conv_data):
        return " | ".join([f"{row['ROLE']}: {row['TEXT']}" for _, row in conv_data.iterrows()])

    def get_conversation_summary(conv_data):
        return {
            'total_messages': len(conv_data),
            'unique_roles': conv_data['ROLE'].nunique(),
            'role_distribution': conv_data['ROLE'].value_counts().to_dict(),
            'conversation_flow': conv_data['ROLE'].tolist()
        }

    for (scenario, run), conversation_data in df.groupby(['scenario', 'run']):
        if len(conversation_data) >= 2:
            key = f"{scenario}_{run}"
            conversations[key] = {
                'key': key,
                'scenario': scenario,
                'run': run,
                'data': conversation_data,
                'text': get_conversation_text(conversation_data),
                'summary': get_conversation_summary(conversation_data)
            }
            
    print(f"   Found {len(conversations)} unique conversations.")
    return conversations

def encode_conversations(model: SentenceTransformer, conversations: Dict[str, Dict], batch_size: int) -> np.ndarray:
    """Encodes all conversation texts into embeddings."""
    conv_texts = [conv['text'] for conv in conversations.values()]
    print(f"🧠 Encoding {len(conv_texts)} conversations using '{CONFIG['model_name']}'...")
    
    embeddings = model.encode(
        conv_texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    return embeddings.cpu().numpy()

# --- Intra-Scenario Analysis (Within the same scenario) ---
def analyze_intra_scenario_similarity(
    conversations: Dict[str, Dict], 
    sim_matrix: np.ndarray, 
    keys: List[str], 
    output_dir: Path
):
    """Analyzes and visualizes similarity within each scenario."""
    print("\n" + "="*80)
    print("📊 Performing Intra-Scenario Analysis (Comparing runs within each scenario)")
    print("="*80)
    
    scenario_dir = output_dir / "intra_scenario_analysis"
    scenario_dir.mkdir(exist_ok=True)
    
    # Group conversations by scenario
    scenarios = defaultdict(list)
    for conv in conversations.values():
        scenarios[conv['scenario']].append(conv['key'])
        
    all_scenario_stats = []

    for scenario, conv_keys in tqdm(scenarios.items(), desc="Analyzing Scenarios"):
        if len(conv_keys) < 2:
            continue

        indices = [keys.index(k) for k in conv_keys]
        scenario_sim_matrix = sim_matrix[np.ix_(indices, indices)]
        
        # Exclude self-similarity for stats
        upper_tri_indices = np.triu_indices_from(scenario_sim_matrix, k=1)
        sim_scores = scenario_sim_matrix[upper_tri_indices]

        if sim_scores.size == 0:
            continue
            
        # --- Statistics ---
        stats = {
            'scenario': scenario,
            'num_conversations': len(conv_keys),
            'avg_similarity': np.mean(sim_scores),
            'std_similarity': np.std(sim_scores),
            'max_similarity': np.max(sim_scores),
            'min_similarity': np.min(sim_scores)
        }
        all_scenario_stats.append(stats)
        
        # --- Visualization: Similarity Heatmap ---
        plt.figure(figsize=(10, 8))
        run_labels = [key.split('_')[-1] for key in conv_keys]
        sns.heatmap(
            scenario_sim_matrix,
            xticklabels=run_labels,
            yticklabels=run_labels,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar=True
        )
        plt.title(f"Intra-Scenario Similarity: {scenario}\nAvg Sim: {stats['avg_similarity']:.2f} ± {stats['std_similarity']:.2f}")
        plt.xlabel("Conversation Run")
        plt.ylabel("Conversation Run")
        plt.tight_layout()
        plt.savefig(scenario_dir / f"{scenario}_similarity_heatmap.png")
        plt.close()

    # Save summary stats to CSV
    if all_scenario_stats:
        stats_df = pd.DataFrame(all_scenario_stats)
        stats_df.to_csv(scenario_dir / "_intra_scenario_summary_stats.csv", index=False)
        print("\n✅ Intra-scenario analysis complete. Heatmaps and summary saved.")

# --- Inter-Scenario Analysis (Between different scenarios) ---
def analyze_inter_scenario_similarity(
    conversations: Dict[str, Dict], 
    sim_matrix: np.ndarray, 
    keys: List[str], 
    output_dir: Path
):
    """Analyzes and visualizes similarity between different scenarios."""
    print("\n" + "="*80)
    print("🌐 Performing Inter-Scenario Analysis (Comparing across different scenarios)")
    print("="*80)
    
    inter_dir = output_dir / "inter_scenario_analysis"
    inter_dir.mkdir(exist_ok=True)
    
    # --- 1. High-level Scenario Similarity Matrix ---
    scenarios = defaultdict(list)
    for i, key in enumerate(keys):
        scenarios[conversations[key]['scenario']].append(i)
        
    scenario_names = list(scenarios.keys())
    num_scenarios = len(scenario_names)
    inter_scenario_matrix = np.zeros((num_scenarios, num_scenarios))

    for i in range(num_scenarios):
        for j in range(num_scenarios):
            indices_i = scenarios[scenario_names[i]]
            indices_j = scenarios[scenario_names[j]]
            
            # Get the sub-matrix of similarities between the two scenarios
            sub_matrix = sim_matrix[np.ix_(indices_i, indices_j)]
            
            if i == j:
                # For diagonal, use intra-scenario similarity (excluding self-sim)
                if len(indices_i) > 1:
                    upper_tri_indices = np.triu_indices(len(indices_i), k=1)
                    inter_scenario_matrix[i, j] = np.mean(sub_matrix[upper_tri_indices])
                else:
                    inter_scenario_matrix[i, j] = 1.0 # Only one conversation
            else:
                inter_scenario_matrix[i, j] = np.mean(sub_matrix)

    plt.figure(figsize=(max(12, num_scenarios * 0.8), max(10, num_scenarios * 0.7)))
    sns.heatmap(
        inter_scenario_matrix,
        xticklabels=scenario_names,
        yticklabels=scenario_names,
        annot=True,
        fmt=".2f",
        cmap="cividis"
    )
    plt.title("Inter-Scenario Average Similarity Matrix\n(Diagonal shows avg. intra-scenario similarity)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(inter_dir / "inter_scenario_similarity_matrix.png")
    plt.close()
    print("   - Generated inter-scenario similarity matrix heatmap.")

    # --- 2. Top Cross-Scenario Similar Pairs ---
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            conv1 = conversations[keys[i]]
            conv2 = conversations[keys[j]]
            if conv1['scenario'] != conv2['scenario']:
                pairs.append({
                    'conv1_key': keys[i],
                    'conv2_key': keys[j],
                    'scenario1': conv1['scenario'],
                    'scenario2': conv2['scenario'],
                    'similarity': sim_matrix[i, j]
                })
    
    if pairs:
        cross_scenario_df = pd.DataFrame(pairs).sort_values(by='similarity', ascending=False)
        cross_scenario_df.to_csv(inter_dir / "cross_scenario_similar_pairs.csv", index=False)
        print(f"   - Found and saved {len(cross_scenario_df)} cross-scenario pairs.")
        
        # --- 3. Scenario Similarity Network Graph ---
        G = nx.Graph()
        top_pairs = cross_scenario_df[cross_scenario_df['similarity'] > CONFIG['similarity_threshold']]
        
        for _, row in top_pairs.iterrows():
            G.add_edge(row['scenario1'], row['scenario2'], weight=row['similarity'])
            
        if G.number_of_edges() > 0:
            plt.figure(figsize=(16, 16))
            pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
            
            # Node size based on number of conversations in that scenario
            node_sizes = [len(scenarios[node]) * 100 + 500 for node in G.nodes()]
            
            # Edge width and color based on similarity weight
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            edge_widths = [w * 10 for w in edge_weights]
            edge_colors = [w for w in edge_weights]
            
            nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=node_sizes,
                    font_size=10, font_weight='bold', edge_color=edge_colors,
                    width=edge_widths, edge_cmap=plt.cm.magma, alpha=0.8)
            
            plt.title("Cross-Scenario Similarity Network", fontsize=20)
            plt.tight_layout()
            plt.savefig(inter_dir / "cross_scenario_network.png")
            plt.close()
            print("   - Generated cross-scenario similarity network graph.")

    print("✅ Inter-scenario analysis complete.")

# --- Main Execution ---
def main():
    """Main function to run the entire analysis pipeline."""
    start_time = time.time()
    setup_plotting_style()
    
    # --- Setup ---
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    device = get_device()
    
    # --- Data Loading and Processing ---
    df = load_and_prepare_data(CONFIG)
    if df.empty:
        return
        
    conversations = group_conversations(df)
    if not conversations:
        print("❌ No conversations found to analyze. Exiting.")
        return
        
    # --- Model Loading and Encoding ---
    model = SentenceTransformer(CONFIG['model_name'], device=device)
    embeddings = encode_conversations(model, conversations, CONFIG['batch_size'])
    
    print("🤝 Computing similarity matrix...")
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()
    
    conv_keys = list(conversations.keys())
    
    # --- Run Analyses ---
    analyze_intra_scenario_similarity(conversations, similarity_matrix, conv_keys, output_dir)
    analyze_inter_scenario_similarity(conversations, similarity_matrix, conv_keys, output_dir)
    
    # --- Final Summary Report ---
    print("\n" + "="*80)
    print("📝 Generating Final Summary Report")
    print("="*80)
    
    report = f"""
# Conversation Similarity Analysis Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Total Conversations Analyzed:** {len(conversations)}
**Total Scenarios:** {df['scenario'].nunique()}

## Intra-Scenario Analysis Summary

This analysis compares different 'runs' of conversations within the same scenario to understand consistency and variance.

- **Key Findings:** See the `_intra_scenario_summary_stats.csv` file for average similarity scores within each scenario.
- **Visualizations:** Heatmaps for each scenario showing run-to-run similarity are saved in `{output_dir / 'intra_scenario_analysis'}`.

## Inter-Scenario Analysis Summary

This analysis compares conversations across different scenarios to find thematic connections and structural similarities.

- **Key Findings:**
  - A high-level similarity matrix (`inter_scenario_similarity_matrix.png`) shows which scenarios are most related on average.
  - A list of the most similar pairs of conversations from *different* scenarios is available in `cross_scenario_similar_pairs.csv`.
  - A network graph (`cross_scenario_network.png`) visually connects scenarios based on the strength of their similarities.

"""
    with open(output_dir / "analysis_summary_report.md", "w") as f:
        f.write(report)
    
    print(f"✅ Analysis complete! All results saved in the '{output_dir}' directory.")
    end_time = time.time()
    print(f"⏱️ Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
