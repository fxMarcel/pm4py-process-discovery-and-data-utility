import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

# Bewertung eines anonymisierten Logs
def evaluate_for_log(original_log, anonymized_log, threshold):
    net, im, fm = pm4py.discover_petri_net_inductive(
        anonymized_log,
        noise_threshold=threshold,
        activity_key="concept:name",
        timestamp_key="time:timestamp",
        case_id_key="concept:name"
    )

    precision = pm4py.precision_token_based_replay(original_log, net, im, fm)
    fitness_result = pm4py.fitness_alignments(original_log, net, im, fm)
    #ggfs. Art der Fitness anpassen ( percentage_of_fitting_traces oder log_fitness)
    fitness = fitness_result["percentage_of_fitting_traces"] / 100.0  # wichtig: Skalierung
    #fitness = fitness_result["log_fitness"]

    f1_score = (
        2 * precision * fitness / (precision + fitness)
        if (precision + fitness) > 0 else 0
    )

    return {
        "fitness": fitness,
        "precision": precision,
        "f1_score": f1_score
    }

# Kombinierte Heatmap zeichnen
def plot_combined_heatmap(matrices, x_labels, y_labels, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["Fitness", "Precision", "F1-Score"]
    cmaps = ["YlGnBu", "YlOrRd", "PuBuGn"]

    for ax, metric, matrix, cmap in zip(axes, metrics, matrices, cmaps):
        sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=x_labels, yticklabels=y_labels,
                    cmap=cmap, vmin=0, vmax=1, ax=ax)
        ax.set_title(f"{metric} Heatmap")
        ax.set_xlabel("L-Wert")
        ax.set_ylabel("K-Wert")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    #Als Input wird ein Ordner benötigt, indem alle TLKC-Anonymisierten Event Logs mit L[Wert]_K[Wert].xes beschriftet sind
    metrics_dir = "Lasagne"
    anonymized_dir = "TLKC_Lasagne"
    os.makedirs(metrics_dir, exist_ok=True)

    # Original-Log laden
    original_log_path = os.path.join(metrics_dir, "20250413_lasagna_event_log_modified.xes")
    original_log = xes_importer.apply(original_log_path)
    threshold = 0.2

    # Anonymisierte Logs laden (Lx_Ky.xes)
    anonymized_files = sorted([
        f for f in os.listdir(anonymized_dir) if re.match(r"L\d+_K\d+\.xes", f)
    ])

    all_results = []
    L_values = set()
    K_values = set()
    result_dict = {}

    for file in anonymized_files:
        match = re.match(r"L(\d+)_K(\d+)\.xes", file)
        if not match:
            continue
        L = int(match.group(1))
        K = int(match.group(2))

        log_path = os.path.join(anonymized_dir, file)
        anonymized_log = xes_importer.apply(log_path)

        print(f"Evaluating L = {L}, K = {K} ({file})")
        result = evaluate_for_log(original_log, anonymized_log, threshold)
        result["L"] = L
        result["K"] = K
        all_results.append(result)

        L_values.add(L)
        K_values.add(K)
        result_dict[(L, K)] = result

    L_sorted = sorted(L_values)
    K_sorted = sorted(K_values)

    def make_matrix(metric):
        matrix = np.zeros((len(K_sorted), len(L_sorted)))
        for i, K in enumerate(K_sorted):
            for j, L in enumerate(L_sorted):
                value = result_dict.get((L, K), {}).get(metric, 0)
                matrix[i][j] = value
        return matrix

    # Heatmaps erzeugen
    fitness_matrix = make_matrix("fitness")
    precision_matrix = make_matrix("precision")
    f1_matrix = make_matrix("f1_score")

    plot_combined_heatmap(
        [fitness_matrix, precision_matrix, f1_matrix],
        x_labels=L_sorted,
        y_labels=K_sorted,
        output_path=os.path.join(metrics_dir, "heatmap_combined.png")
    )

    # Ergebnisse speichern
    with open(os.path.join(metrics_dir, "heatmap_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nBenchmark abgeschlossen!")
    print("Heatmap gespeichert unter: Lasagne/heatmap_combined.png")
    print("Ergebnisse gespeichert in: Lasagne/heatmap_results.json")

if __name__ == "__main__":
    main()
