import os
import json
import pm4py
import matplotlib.pyplot as plt
import pm4py.algo.filtering
import pm4py.algo.filtering.dfg
import pm4py.algo.filtering.dfg.dfg_filtering
from pm4py.objects.log.importer.xes import importer as xes_importer


log_dir = "Klein"
original_log_path = os.path.join(log_dir, "20250414_klein_event_log.xes")
original_log = xes_importer.apply(original_log_path)

dfg = pm4py.discover_dfg(original_log)

activities = set()
for (src, tgt) in dfg[0].keys():
    activities.add(src)
    activities.add(tgt)

print(activities)

dfg_filtered = pm4py.algo.filtering.dfg.dfg_filtering.clean_dfg_based_on_noise_thresh(dfg[0], activities, 0.2)

#Hier ist die angepasste Zeile mit Start-/End-Activities
net, im, fm = pm4py.objects.conversion.dfg.variants.to_petri_net_invisibles_no_duplicates.apply(
    dfg_filtered,
    parameters={
        "start_activities": dfg[1],
        "end_activities": dfg[2]

    }
)
pm4py.view_petri_net(net,im,fm)


# Berechne Precision und Fitness (über Alignments)
precision = pm4py.precision_token_based_replay(original_log, net, im, fm)
fitness_result = pm4py.fitness_token_based_replay(original_log, net, im, fm)
# hier ggfs. auswählen, welche fitness verwendet werden soll
fitness = fitness_result["percentage_of_fitting_traces"] / 100.0  
#fitness = fitness_result["log_fitness"]

# F1-Score berechnen
f1_score = 2 * (fitness * precision) / (fitness + precision) if (fitness + precision) > 0 else 0

# JSON-Daten vorbereiten

metrics_output = {
    "name": "Spaghetti",
    "typ":"percentage_of_fitting_traces",
    "fitness": fitness,
    "precision": precision,
    "f1_score": f1_score
}

# In Datei speichern
with open("Spaghetti_DFG_to_Petri.json", "w") as f:
    json.dump(metrics_output, f, indent=4)

print("\nMetriken gespeichert")
