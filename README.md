# TOMG-Bench
## Text-based Open Molecule Generation Language Model Benchmark

A benchmark for evaluating LLMs' performance on **Text-based Open Molecule Generation** tasks.

Authors: Jiatong Li*, Junxian Li*, Yunqing Liu, Dongzhan Zhou, and Qing Li 

Arxiv: [https://arxiv.org/abs/xxx](https://arxiv.org/abs/xxx)
Huggingface Dataset: [https://huggingface.co/datasets/Duke-de-Artois/TOMG-Bench](https://huggingface.co/datasets/Duke-de-Artois/TOMG-Bench)
PaperWithCode: []()

Benchmark Page: []()

Leaderboard: []()

## Introduction
### Dataset Categorization
This repository contains the code for the TOMG-Bench benchmark, which is a benchmark for evaluating LLMs' performance on Text-based Open Molecule Generation tasks. The benchmark consists of three main tasks. Each task is further divided into three subtasks, and each subtask is composed of 5000 data samples. Below is the dataset categorization:
* [MolCustom](./data/benchmarks/open_generation/MolCustom/readme.md)
  - AtomNum
  - FunctionalGroup
  - BondNum
* [MolEdit](./data/benchmarks/open_generation/MolEdit/readme.md)
  - AddComponent
  - DelComponent
  - SubComponent
* [MolOpt](./data/benchmarks/open_generation/MolOpt/readme.md)
  - LogP
  - MR
  - QED

### Metrics
We adopt different evaluation metrics for different tasks. The evaluation metrics for each subtask are described in the corresponding subtask's README file.

The leader board is based on the weighted average accuracy metric, which is discussed in our paper.

### Usage
1. To query propietary models, please refer to the [query_openai](./query_openai.py).
2. To evaluate the performance of an open-source general LLM, please refer to the [run_query_vllm](./run_query_vllm.bash).
3. To evaluate the performance of a ChEBI-20 fine-tuned LLM, please refer to the [run_query_biot5](./run_query_biot5.bash) and [run_query_molt5](./run_query_molt5.bash).
4. To train on our OpenMolIns dataset, please refer to the [train](./run_train.bash).
5. To evaluate your model on our benchmark, please refer to the [run_query_template](./run_query_template.bash).

### Submit Your Model

If your model achieves amazing performance on our benchmark and you want to update the leaderboard, please send us your results (including raw files) via our emails. We will help you update the leaderboard once we verified the results.

## Reference
```
```
