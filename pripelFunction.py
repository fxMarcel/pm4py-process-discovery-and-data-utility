import os
import re
import json
import matplotlib.pyplot as plt
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

def evaluate_for_log(original_log, anonymized_log, threshold):
    net, im, fm = pm4py.discover_petri_net_inductive(
        anonymized_log,
        noise_threshold=threshold,
        activity_key="concept:name",
        timestamp_key="time:timestamp",
        case_id_key="concept:name"
    )

    precision = pm4py.precision_token_based_replay(original_log, net, im, fm)
    fitness_result = pm4py.fitness_token_based_replay(original_log, net, im, fm)
    fitness = fitness_result["percentage_of_fitting_traces"] / 100
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

def plot_metrics(results, output_path):
    epsilons = [r["epsilon"] for r in results]
    fitnesses = [r["fitness"] for r in results]
    precisions = [r["precision"] for r in results]
    f1_scores = [r["f1_score"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, fitnesses, marker="o", label="Fitness")
    plt.plot(epsilons, precisions, marker="s", label="Precision")
    plt.plot(epsilons, f1_scores, marker="^", label="F1-Score")

    plt.xlabel("ε (Epsilon)")
    plt.ylabel("Score (0–1)")
    plt.title("Fitness, Precision & F1-Score über Epsilon (PRIPEL)")
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    #Hierfür muss wieder ein Ordner mit den Input XES-Dateien vorhanden sein, das Benennungsschema in diesem Fall war 20250414_klein_event_log_epsilon_([0-9.]+)_k_1_anonymized\.xes, sodass Epsilon am Dateinamen erkannt werden kann
    metrics_dir = "Klein"
    anonymized_dir = "PRIPEL_Klein_Input"
    os.makedirs(metrics_dir, exist_ok=True)

    original_log_path = os.path.join(metrics_dir, "20250414_klein_event_log.xes")
    original_log = xes_importer.apply(original_log_path)
    threshold = 0.2

    anonymized_files = sorted([
        f for f in os.listdir(anonymized_dir) if f.endswith(".xes")
    ])

    all_results = []

    for file in anonymized_files:
        # Beispielname: 20250414_klein_event_log_epsilon_0.1_k_1_anonymized.xes
        match = re.match(r"20250414_klein_event_log_epsilon_([0-9.]+)_k_1_anonymized\.xes", file)
        if not match:
            continue

        epsilon = float(match.group(1))
        log_path = os.path.join(anonymized_dir, file)
        anonymized_log = xes_importer.apply(log_path)

        print(f"🔍 Evaluating ε = {epsilon} ({file})")
        result = evaluate_for_log(original_log, anonymized_log, threshold)
        result["epsilon"] = epsilon
        all_results.append(result)

    # Nach aufsteigendem Epsilon sortieren
    all_results.sort(key=lambda r: r["epsilon"])

    # Ergebnisse speichern
    json_path = os.path.join(metrics_dir, "pripel_epsilon_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)

    # Plot erzeugen
    plot_path = os.path.join(metrics_dir, "pripel_epsilon_plot.png")
    plot_metrics(all_results, plot_path)

    print("\nPRIPEL-Evaluierung abgeschlossen!")
    print(f"Ergebnisse gespeichert in: {json_path}")
    print(f"Plot gespeichert unter:    {plot_path}")

if __name__ == "__main__":
    main()
