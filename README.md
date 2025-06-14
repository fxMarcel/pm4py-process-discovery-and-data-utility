# PM4Py Process Discovery and Data Utility

This repository comprises a collection of standalone Python scripts that leverage the [PM4Py](https://github.com/process-intelligence-solutions/pm4py) library for process discovery and data utility tasks.  
The data utility results are visualized using Matplotlib and Seaborn.

## Overview

The scripts in this repository perform various process mining operations, including:

- **Process Discovery**: Utilizing algorithms like the Inductive Miner to derive process models from event logs.
- **Data Utility Analysis and Visualization**: Assessing the data utility of anonymized event logs.

Each script is designed to be executed independently, allowing for modular analysis and experimentation.

## Repository Structure

- `DFGMatplot.py`: Generates Directly-Follows Graphs (DFGs), measures data utility measures and visualizes them using Matplotlib.
- `DFGToPetri.py`: Converts DFGs into Petri nets.
- `LogSampling.py`: Performs sampling on event logs.
- `TLKCFunctionK.py` & `TLKCHeatmap.py`: Analyze and visualize (with TLKC anonymized) event logs and visualize the data utility results with matplot as heatmap / function.
- `UtilityFunctionNachThreshold.py`: Applies utility functions based on specified thresholds to assess data quality.
- `inductiveMinerManuell.py`: Manually applies the Inductive Miner algorithm for process discovery.
- `pripelFunction.py`: Analyze and visualize (with PRIPEL anonymized) event logs and visualize the data utility results with matplot as function.
- `requirements.txt`: Lists all Python dependencies required to run the scripts.

