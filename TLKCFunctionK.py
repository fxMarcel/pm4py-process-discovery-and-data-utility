import os
import re
import json
import pm4py
import matplotlib.pyplot as plt
from pm4py.objects.log.importer.xes import importer as xes_importer

# Evaluierung für einen anonymisierten Log
def evaluate_for_log(original_log, anonymized_log, threshold):
    net, im, fm = pm4py.discover_petri_net_inductive(
        anonymized_log,
        noise_threshold=threshold,
        activity_key="concept:name",
        timestamp_key="time:timestamp",
        case_id_key="concept:name"
    )

    # Precision & Fitness
    precision = pm4py.precision_token_based_replay(original_log, net, im, fm)
    fitness_result = pm4py.fitness_token_based_replay(original_log, net, im, fm)
    #ggfs. Art der Fitness anpassen ( percentage_of_fitting_traces oder log_fitness)
    fitness = fitness_result["percentage_of_fitting_traces"] / 100.0 
    #fitness = fitness_result["log_fitness"]

    # F1-Score
    f1_score = (
        2 * precision * fitness / (precision + fitness)
        if (precision + fitness) > 0 else 0
    )

    return {
        "fitness": fitness,
        "precision": precision,
        "f1_score": f1_score
    }

# Zeichne Diagramm
def plot_results(results, output_path):
    # Nach K-Werten sortieren
    results = sorted(results, key=lambda r: r["K"])

    K_values = [r["K"] for r in results]
    fitnesses = [r["fitness"] for r in results]
    precisions = [r["precision"] for r in results]
    f1_scores = [r["f1_score"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(K_values, fitnesses, marker="o", label="Fitness")
    plt.plot(K_values, precisions, marker="s", label="Precision")
    plt.plot(K_values, f1_scores, marker="^", label="F1-Score")

    for x, y in zip(K_values, fitnesses):
        plt.text(x, y + 0.02, f"{y:.2f}", ha='center', fontsize=8, color='blue')
    for x, y in zip(K_values, precisions):
        plt.text(x, y + 0.02, f"{y:.2f}", ha='center', fontsize=8, color='green')
    for x, y in zip(K_values, f1_scores):
        plt.text(x, y + 0.02, f"{y:.2f}", ha='center', fontsize=8, color='black')

    plt.xlabel("K-Wert")
    plt.ylabel("Score (0-1)")
    plt.title("Fitness, Precision & F1-Score über verschiedene K-Werte")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Extrahiere K-Wert aus Dateinamen, z. B. "[25]"
def extract_k_from_filename(filename):
    match = re.search(r'\[(\d+)\]', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"K-Wert konnte aus dem Dateinamen '{filename}' nicht extrahiert werden.")

def main():
    metrics_dir = "Klein"
    anonymized_dir = "TLKC_K_Klein"
    os.makedirs(metrics_dir, exist_ok=True)

    # Log-Pfad Original
    original_log_path = os.path.join(metrics_dir, "20250414_klein_event_log.xes")
    original_log = xes_importer.apply(original_log_path)

    threshold = 0.2

    # Anonymisierte Logs laden
    anonymized_files = sorted([
        f for f in os.listdir(anonymized_dir)
        if f.endswith(".xes")
    ])

    all_results = []

    for file in anonymized_files:
        log_path = os.path.join(anonymized_dir, file)
        anonymized_log = xes_importer.apply(log_path)

        try:
            K_value = extract_k_from_filename(file)
        except ValueError as e:
            print(f"⚠️  {e}")
            continue

        print(f"Evaluating K = {K_value} ({file})")
        result = evaluate_for_log(original_log, anonymized_log, threshold)
        result["K"] = K_value
        print(f"Fitness: {result['fitness']:.4f}, Precision: {result['precision']:.4f}, F1: {result['f1_score']:.4f}")
        all_results.append(result)

    # Ergebnisse speichern
    json_path = os.path.join(metrics_dir, "benchmarking_results_TLKC_K.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)

    # Plot speichern
    plot_path = os.path.join(metrics_dir, "benchmarking_plot_TLKC_K.png")
    plot_results(all_results, plot_path)

    print("\nBenchmark abgeschlossen!")
    print(f"Ergebnisse gespeichert in: {json_path}")
    print(f"Plot gespeichert unter:    {plot_path}")

if __name__ == "__main__":
    main()
