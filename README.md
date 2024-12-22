# TOMG-Bench
## Text-based Open Molecule Generation Benchmark

Authors: Jiatong Li*, Junxian Li*, Yunqing Liu, Dongzhan Zhou, and Qing Li （* Equal Contribution)

Arxiv: [https://arxiv.org/abs/2412.14642](https://arxiv.org/abs/2412.14642)  
Huggingface Dataset: [https://huggingface.co/datasets/Duke-de-Artois/TOMG-Bench](https://huggingface.co/datasets/Duke-de-Artois/TOMG-Bench)  
PaperWithCode: [https://paperswithcode.com/dataset/tomg-bench](https://paperswithcode.com/dataset/tomg-bench)

Benchmark Page: []()

Leaderboard: []()

## Introduction
In this paper, we propose **Text-based Open Molecule Generation Benchmark (TOMG-Bench)**, the first benchmark to evaluate the open-domain molecule generation capability of LLMs. TOMG-Bench encompasses a dataset of three major tasks: molecule editing (MolEdit), molecule optimization (MolOpt), and customized molecule generation (MolCustom). Each task further contains three subtasks, with each subtask comprising 5,000 test samples. Given the inherent complexity of open molecule generation, we have also developed an automated evaluation system that helps measure both the quality and the accuracy of the generated molecules. Our comprehensive benchmarking of 25 LLMs reveals the current limitations and potential areas for improvement in text-guided molecule discovery. Furthermore, with the assistance of OpenMolIns, a specialized instruction tuning dataset proposed for solving challenges raised by TOMG-Bench, Llama3.1-8B could outperform all the open-source general LLMs, even surpassing GPT-3.5-turbo by 46.5% on TOMG-Bench.

### Pipeline  
![image](https://github.com/user-attachments/assets/bb9638aa-922c-478b-b5d8-0d33c00f89e3)

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

The leaderboard is based on the weighted average accuracy metric, which is discussed in our paper.

### Usage
1. To query proprietary models, please refer to the [query_openai](./query_openai.py).
2. To evaluate the performance of an open-source general LLM, please refer to the [run_query_vllm](./run_query_vllm.bash).
3. To evaluate the performance of a ChEBI-20 fine-tuned LLM, please refer to the [run_query_biot5](./run_query_biot5.bash) and [run_query_molt5](./run_query_molt5.bash).
4. To train on our OpenMolIns dataset, please refer to the [train](./run_train.bash).
5. To evaluate your model on our benchmark, please refer to the [run_query_template](./run_query_template.bash).

### Submit Your Model

If your model achieves amazing performance on our benchmark and you want to update the leaderboard, please send us your results (including raw files) via our emails. We will help you update the leaderboard once we have verified the results.

## Reference
```
@misc{li2024tomgbenchevaluatingllmstextbased,
      title={TOMG-Bench: Evaluating LLMs on Text-based Open Molecule Generation}, 
      author={Jiatong Li and Junxian Li and Yunqing Liu and Dongzhan Zhou and Qing Li},
      year={2024},
      eprint={2412.14642},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.14642}, 
}
```
