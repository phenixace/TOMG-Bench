# TOMG-Bench
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2412.14542-B31B1B.svg)](https://arxiv.org/abs/2412.14642)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/Duke-de-Artois/TOMG-Bench)
[![PaperWithCode](https://img.shields.io/badge/PWC-Dataset-blue)](https://paperswithcode.com/dataset/tomg-bench)

## Text-based Open Molecule Generation Benchmark

Authors: Jiatong Li*, Junxian Li*, Yunqing Liu, Dongzhan Zhou, and Qing Li （* Equal Contribution)

* Arxiv: [https://arxiv.org/abs/2412.14642](https://arxiv.org/abs/2412.14642)  
* Huggingface Dataset: [https://huggingface.co/datasets/Duke-de-Artois/TOMG-Bench](https://huggingface.co/datasets/Duke-de-Artois/TOMG-Bench)  
* PaperWithCode: [https://paperswithcode.com/dataset/tomg-bench](https://paperswithcode.com/dataset/tomg-bench)
* Project Page: [https://phenixace.github.io/tomgbench/](https://phenixace.github.io/tomgbench/)

## Introduction
In this paper, we propose **Text-based Open Molecule Generation Benchmark (TOMG-Bench)**, the first benchmark to evaluate the open-domain molecule generation capability of LLMs. TOMG-Bench encompasses a dataset of three major tasks: molecule editing (MolEdit), molecule optimization (MolOpt), and customized molecule generation (MolCustom). Each task further contains three subtasks, with each subtask comprising 5,000 test samples. Given the inherent complexity of open molecule generation, we have also developed an automated evaluation system that helps measure both the quality and the accuracy of the generated molecules. Our comprehensive benchmarking of 25 LLMs reveals the current limitations and potential areas for improvement in text-guided molecule discovery. Furthermore, with the assistance of OpenMolIns, a specialized instruction tuning dataset proposed for solving challenges raised by TOMG-Bench, Llama3.1-8B could outperform all the open-source general LLMs, even surpassing GPT-3.5-turbo by 46.5% on TOMG-Bench.

### Leaderboard
| Model                           | #Parameters | A̅cc (%) | wA̅cc (%) |
|---------------------------------|---------------|---------|----------|
| Claude-3.5 (Anthropic, 2024b)   | -             | 51.10   | 35.92    |
| Gemini-1.5-pro (Deepmind, 2024) | -             | 52.25   | 34.80    |
| GPT-4-turbo (Achiam et al., 2023)| -            | 50.74   | 34.23    |
| GPT-4o (Achiam et al., 2023)    | -             | 49.08   | 32.29    |
| Claude-3 (Anthropic, 2024a)     | -             | 46.14   | 30.47    |
| OpenMolIns-large (Llama-3.1-8B) | 8B             | 43.1    | 27.22    |
| OpenMolIns-xlarge (Galactica-125M)| 125M     | 44.48   | 25.73    |
| Llama3-70B-Instruct (Int4) (Dubey et al., 2024) | 70B | 38.54 | 23.93 |
| OpenMolIns-large (Galactica-125M)| 125M       | 39.28   | 23.42    |
| OpenMolIns-medium (Galactica-125M)| 125M     | 34.54   | 19.89    |
| GPT-3.5-turbo (Achiam et al., 2023)| -          | 28.93   | 18.58    |
| OpenMolIns-small (Galactica-125M)| 125M       | 24.17   | 15.18    |
| Llama3.1-8B-Instruct (Dubey et al., 2024) | 8B | 26.26 | 14.09 |
| Llama3-8B-Instruct (Dubey et al., 2024) | 8B | 26.40 | 13.75 |
| chatglm-9B (GLM et al., 2024)    | 9B            | 18.50   | 13.13(7) |
| OpenMolIns-light (Galactica-125M)| 125M       | 20.95   | 13.13(6) |
| OpenMolIns-large (Llama3.2-1B)  | 1B             | 14.11   | 8.10     |
| yi-1.5-9B (Young et al., 2024)   | 9B            | 14.10   | 7.32     |
| Mistral-7B-Instruct-v0.2 (Jiang et al., 2023) | 7B | 11.17 | 4.81 |
| BioT5-base (Pei et al., 2023)    | 250M        | 24.19   | 4.21     |
| MolT5-large (Edwards et al., 2022)| 780M         | 23.11   | 2.89     |
| Llama-3.1-1B-Instruct (Dubey et al., 2024) | 1B | 3.95 | 1.99 |
| MolT5-base (Edwards et al., 2022) | 250M           | 11.11   | 1.30(0)  |
| MolT5-small (Edwards et al., 2022)| 80M           | 11.55   | 1.29(9)  |
| Qwen2-7B-Instruct (Yang et al., 2024) | 7B | 0.18 | 0.15     |



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
2. To evaluate the performance of an open-source general LLM, please refer to the [run_query_8b](./run_query_copy_8B.bash). Remember to use your own model path!    
3. To evaluate the performance of a ChEBI-20 fine-tuned LLM, please refer to the [run_query_biot5](./run_query_biot5.bash) and [run_query_molt5](./run_query_molt5.bash).
4. To train on our OpenMolIns dataset, please refer to the [train](./run_train.bash).
5. To evaluate your model on our benchmark, please refer to the [run_query_template](./run_query_template.bash).
6. After generating the csv file with answers from various models, please run the following command to get the score. If you set ```calc_novelty``` to True, it's necessary for you to assign a GPU for this evaluation.  
```python evaluate.py --name your_models_name --task xxx(MolCustom, MolOpt or MolEdit) --subtask xxx```


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
