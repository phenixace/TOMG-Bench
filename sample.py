# TOD: sample 5000 molecules from the ZINC and PubChem datasets
import pandas as pd

# Load the dataset
data = pd.read_csv("~/projects/OMG-LLMBench/data/sources/zinc250k/zinc250k_selfies.csv")

print(data.head())

# # Sample 5000 molecules
sample = data.sample(n=5000)

# # Save the sample
sample.to_csv("~/projects/OMG-LLMBench/data/benchmarks/open_generation/MolEdit/AddComponent/test_raw.csv", index=False)

# # Sample 8000 molecules
sample = data.sample(n=8000)

# # Save the sample
sample.to_csv("~/projects/OMG-LLMBench/data/benchmarks/open_generation/MolEdit/DelComponent/test_raw.csv", index=False)

# Sample 8000 molecules
sample = data.sample(n=12000)

# Save the sample
sample.to_csv("~/projects/OMG-LLMBench/data/benchmarks/open_generation/MolEdit/SubComponent/test_raw.csv", index=False)

# # Sample 5000 molecules
sample = data.sample(n=5000)

# # Save the sample
sample.to_csv("~/projects/OMG-LLMBench/data/benchmarks/open_generation/MolOpt/LogP/test_raw.csv", index=False)

# # Sample 5000 molecules
sample = data.sample(n=5000)

# # Save the sample
sample.to_csv("~/projects/OMG-LLMBench/data/benchmarks/open_generation/MolOpt/MR/test_raw.csv", index=False)

# # Sample 5000 molecules
sample = data.sample(n=5000)

# # Save the sample
sample.to_csv("~/projects/OMG-LLMBench/data/benchmarks/open_generation/MolOpt/QED/test_raw.csv", index=False)