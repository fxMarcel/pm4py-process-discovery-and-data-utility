import os
import json
import pm4py
import matplotlib.pyplot as plt
from pm4py.objects.log.importer.xes import importer as xes_importer

# Log einlesen
log = xes_importer.apply("20250414_spaghetti_event_log_v2_modified.xes")

# Varianten zählen
filter_log = pm4py.filter_variants_top_k(log,100)
pm4py.objects.log.exporter.xes.exporter.apply(filter_log,"spaghetti_top_100.xes")


