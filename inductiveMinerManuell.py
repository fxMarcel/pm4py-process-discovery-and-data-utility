import os
import json
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.inductive.variants import imf
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.conversion.process_tree.converter import Variants as PTVariants


def main():
    # Ordnername
    metrics_dir = "Theorie_Kapitel"
    os.makedirs(metrics_dir, exist_ok=True)

    # Pfade zu den Logs
    original_log_path = os.path.join(metrics_dir, "20250414_klein_event_log.xes")
    anonymized_log_path = os.path.join(metrics_dir, "20250327_Beispiellog_v4.xes")

    # Logs einlesen
    original_log = xes_importer.apply(original_log_path)
    anonymized_log = xes_importer.apply(anonymized_log_path)

    df = pm4py.convert_to_dataframe(anonymized_log)

    # Nur die relevanten Spalten Case ID und Activity auswählen
    df_cases_activities = df[['case:concept:name', 'concept:name']]

    # Gruppierung der Aktivitäten pro Case
    df_grouped = df_cases_activities.groupby('case:concept:name')['concept:name'].apply(lambda x: " → ".join(x)).reset_index()
    df_grouped.columns = ['Case ID', 'Activities']
    print(df_grouped)
    df_grouped.to_csv('beispiellog.csv')


    df = df[['case:concept:name', 'concept:name', 'time:timestamp', 'Kosten', 'org:resource']]
    df.columns = ['case_id', 'activity', 'timestamp', 'costs', 'org:resource']
    df
    print(df.head())

    
    net, im, fm = pm4py.discover_petri_net_inductive(anonymized_log, True, 0.2, "concept:name", "time:timestamp", "concept:name")
    pm4py.view_petri_net(net,im,fm)
    
    # Precision (Token-based Replay)
    precision = pm4py.precision_token_based_replay(original_log, net, im, fm)

    # Fitness (Alignment-based)
    fitness = pm4py.fitness_alignments(original_log, net, im, fm)

    # Ergebnisse speichern
    metrics = {
        "Fitness": fitness,
        "Precision": precision
    }

    with open(os.path.join(metrics_dir, "evaluation_results.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Ausgabe
    print(" Metriken erfolgreich berechnet und gespeichert unter 'Metrics/evaluation_results.json'")
    print(f"Fitness:               {fitness}")
    print(f"Precision:             {precision}")
    



if __name__ == "__main__":
    main()
