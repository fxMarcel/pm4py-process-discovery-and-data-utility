import os
import json
import pm4py
import matplotlib.pyplot as plt
from pm4py.objects.log.importer.xes import importer as xes_importer

# Evaluierung für einen Noise-Wert
def evaluate_for_threshold(threshold, original_log, anonymized_log):
    net, im, fm = pm4py.discover_petri_net_inductive(
        anonymized_log, 
        noise_threshold=threshold, 
        activity_key="concept:name", 
        timestamp_key="time:timestamp", 
        case_id_key="concept:name"
    )

    # Precision & Fitness
    precision = pm4py.precision_token_based_replay(original_log, net, im, fm)
    fitness_result = pm4py.fitness_alignments(original_log, net, im, fm)
    #ggfs. Art der Fitness anpassen ( percentage_of_fitting_traces oder log_fitness)
    fitness = fitness_result["percentage_of_fitting_traces"] / 100
    #fitness = fitness_result["log_fitness"]

    # F1-Score
    f1_score = (
        2 * precision * fitness / (precision + fitness)
        if (precision + fitness) > 0 else 0
    )

    return {
        "threshold": threshold,
        "fitness": fitness,
        "precision": precision,
        "f1_score": f1_score
    }

# Zeichne Diagramm
def plot_results(results, output_path):
    thresholds = [r["threshold"] for r in results]
    fitnesses = [r["fitness"] for r in results]
    precisions = [r["precision"] for r in results]
    f1_scores = [r["f1_score"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, fitnesses, marker="o", label="Fitness")
    plt.plot(thresholds, precisions, marker="s", label="Precision")
    plt.plot(thresholds, f1_scores, marker="^", label="F1-Score")

    # Werte direkt an die Punkte schreiben
    for x, y in zip(thresholds, fitnesses):
        plt.text(x, y + 0.02, f"{y:.2f}", ha='center', fontsize=8, color='blue')
    for x, y in zip(thresholds, precisions):
        plt.text(x, y + 0.02, f"{y:.2f}", ha='center', fontsize=8, color='green')
    for x, y in zip(thresholds, f1_scores):
        plt.text(x, y + 0.02, f"{y:.2f}", ha='center', fontsize=8, color='black')

    plt.xlabel("IMF Noise Threshold")
    plt.ylabel("Score (0-1)")
    plt.title("Fitness, Precision & F1-Score über Noise Thresholds")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    lasagne_dir = "Lasagne"
    os.makedirs(lasagne_dir, exist_ok=True)

    # Log-Pfade
    original_log_path = os.path.join(lasagne_dir, "20250413_lasagna_event_log_modified.xes")
    anonymized_log_path = os.path.join(lasagne_dir, "20250413_lasagna_event_log_modified.xes")

    # Logs laden
    original_log = xes_importer.apply(original_log_path)
    anonymized_log = xes_importer.apply(anonymized_log_path)

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    all_results = []

    for threshold in thresholds:
        print(f"Evaluating threshold = {threshold}")
        result = evaluate_for_threshold(threshold, original_log, anonymized_log)
        print(f"Fitness: {result['fitness']:.4f}, Precision: {result['precision']:.4f}, F1: {result['f1_score']:.4f}")
        all_results.append(result)

    # JSON speichern
    json_path = os.path.join(lasagne_dir, "TLKC_benchmarking_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)

    # Plot speichern
    plot_path = os.path.join(lasagne_dir, "TLKC_benchmarking_plot.png")
    plot_results(all_results, plot_path)

    print("\nBenchmark abgeschlossen!")
    print(f"Ergebnisse gespeichert in: {json_path}")
    print(f"Plot gespeichert unter:    {plot_path}")

if __name__ == "__main__":
    main()
