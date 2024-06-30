import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_json(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


# Load JSON files
bp_33 = load_json(Path(__file__).parent.joinpath('rbp_batch_size_33.json'))
bp_100 = load_json(Path(__file__).parent.joinpath('rbp_batch_size_100.json'))
pc_33 = load_json(Path(__file__).parent.joinpath('pcc_batch_size_33.json'))
pc_100 = load_json(Path(__file__).parent.joinpath('pcc_batch_size_100.json'))


# Extract test__classification_error and add additional info
def extract_errors(data, model_type, batch_size):
    return [{'iteration': i, 'error': obj['test__classification_error'], 'model': model_type, 'batch_size': batch_size}
            for i, obj in enumerate(data)]


bp_33_errors = extract_errors(bp_33, 'BP', 33)
bp_100_errors = extract_errors(bp_100, 'BP', 100)
pc_33_errors = extract_errors(pc_33, 'PC', 33)
pc_100_errors = extract_errors(pc_100, 'PC', 100)

# Combine all data into a DataFrame
all_errors = bp_33_errors + bp_100_errors + pc_33_errors + pc_100_errors
df = pd.DataFrame(all_errors)

# Plotting
sns.set(style='whitegrid')
plt.figure(figsize=(14, 8))

# Use different line styles for batch sizes
line_styles = {33: '-', 100: '--'}

# Create the plot
sns.lineplot(data=df, x='iteration', y='error', hue='model', style='batch_size', palette='deep', dashes=line_styles)

# Customize the plot
plt.title('Test Classification Error over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Test Classification Error')
plt.legend(title='Model / Batch Size')

# Save the plot with 150 DPI
plt.savefig('classification_error_plot.png', dpi=150)
