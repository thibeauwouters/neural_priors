import json
import numpy as np
import matplotlib.pyplot as plt

filename = "all_bayes_factors.json"

# Load it
with open(filename, "r") as f:
    data = json.load(f)
    
log_evidence_errors = data["log_evidence_errors"]

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(log_evidence_errors, bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.title('Histogram of Log Evidence Errors')
plt.xlabel('Log Evidence Error in natural log')
plt.ylabel('Frequency')
# Save the histogram
plt.savefig("log_evidence_errors_histogram.png")
plt.close()