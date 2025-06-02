import json
import matplotlib.pyplot as plt
import numpy as np

# JSON-Datei laden
with open("DFG_to_Petri_Gesamt.json", "r") as f:
    data = json.load(f)

# Gruppierung der Daten nach Logs und Typen
logs = ['Spaghetti', 'Lasagne', 'Klein']
types = ['log_fitness', 'percentage_of_fitting_traces']
metrics = ['fitness', 'precision', 'f1_score']

# Farben in unterschiedlichen Blautönen
colors = {
    'fitness': '#4c78a8',     # dunkles Blau
    'precision': '#6baed6',   # helleres Blau
    'f1_score': '#9ecae1'     # noch helleres Blau
}

# Plot vorbereiten
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.2
x = np.arange(len(logs) * len(types))

# Werte sammeln
values = {'fitness': [], 'precision': [], 'f1_score': []}
x_labels = []

for log in logs:
    for typ in types:
        entry = next(item for item in data if item['name'] == log and item['typ'] == typ)
        for metric in metrics:
            values[metric].append(entry[metric])
        x_labels.append(f"{log}\n{typ}")

# Balken zeichnen + Wertebeschriftung
offsets = {'fitness': -bar_width, 'precision': 0, 'f1_score': bar_width}
for metric in metrics:
    bars = ax.bar(x + offsets[metric], values[metric], width=bar_width, label=metric, color=colors[metric])
    # Werte über Balken anzeigen
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # vertikaler Abstand
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# Achsen und Beschriftung
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.set_ylabel("Score")
ax.set_title("Fitness, Precision und F1-Score je Log und Fit-Typ")
ax.legend(title="Metrik")
plt.tight_layout()
plt.show()
