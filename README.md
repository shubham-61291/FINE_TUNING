<p align="center">
  <a href="https://github.com/shubham-61291/FINE_TUNING"><img src="https://img.shields.io/badge/TRANSFORMER-FINE--TUNING_LAB-20232A?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E" alt="Transformer Fine-Tuning Laboratory" /></a>
</p>

<h1 align="center">Transformer Fine-Tuning Laboratory</h1>

<p align="center"><strong>A controlled comparison of head-only fine-tuning, last-layer fine-tuning, LoRA, and QLoRA for IMDb sentiment classification.</strong></p>

<p align="center"><a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-Training-3776AB?style=for-the-badge&amp;logo=python&amp;logoColor=white" alt="Python training" /></a> &nbsp; <a href="https://huggingface.co/docs/transformers/index"><img src="https://img.shields.io/badge/Hugging_Face-Transformers-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=black" alt="Hugging Face Transformers" /></a> &nbsp; <a href="https://huggingface.co/datasets/stanfordnlp/imdb"><img src="https://img.shields.io/badge/Dataset-IMDb-4A4A4A?style=for-the-badge&amp;logo=imdb&amp;logoColor=white" alt="IMDb dataset" /></a> &nbsp; <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-238636?style=for-the-badge&amp;logo=opensourceinitiative&amp;logoColor=white" alt="MIT License" /></a></p>

<p align="center"><strong>License:</strong> <a href="LICENSE">MIT License</a> · <strong>Experiments:</strong> 4 · <strong>Task:</strong> Binary sentiment classification</p>

This repository contains four focused experiments that adapt pretrained Transformer sequence-classification models to binary movie-review sentiment. The experiments compare how much of a pretrained network is updated, how parameter-efficient methods change the trainable parameter set, and how evaluation metrics look before and after training.

The intent is educational and comparative: each directory includes its training script, method-specific README, and result plots. The training scripts are exported from Google Colab notebooks and use Hugging Face Transformers, Datasets, Evaluate, and the relevant parameter-efficient fine-tuning libraries.

---

## Contents

- [Experiment map](#experiment-map)
- [Repository layout](#repository-layout)
- [What the experiments compare](#what-the-experiments-compare)
- [Common training and evaluation protocol](#common-training-and-evaluation-protocol)
- [Results and chart interpretation](#results-and-chart-interpretation)
- [Environment and installation](#environment-and-installation)
- [Inference concept](#inference-concept)
- [Reproducibility and customization](#reproducibility-and-customization)
- [Scope and interpretation](#scope-and-interpretation)
- [Author](#author)
- [License](#license)


**Quick navigation:** [Head-only](NEW_HEAD/README.md) · [Last-layer](LAST_LAYER/README.md) · [LoRA](PEFT_LORA/README.md) · [QLoRA](PEFT_QLORA/README.md) · [MIT License](LICENSE)

## Experiment map

| Experiment | Directory | Base checkpoint | What is trained | Main efficiency lever |
|---|---|---|---|---|
| New classification head | [`NEW_HEAD/`](NEW_HEAD/README.md) | `distilbert-base-uncased` | Newly initialized classification head; backbone frozen | Minimal trainable parameter set |
| Last Transformer block + head | [`LAST_LAYER/`](LAST_LAYER/README.md) | `distilbert-base-uncased` | Final Transformer block and classification head | Partial backbone adaptation |
| LoRA | [`PEFT_LORA/`](PEFT_LORA/README.md) | `bert-base-uncased` | LoRA adapters and task head according to PEFT sequence-classification wrapping | Low-rank trainable updates |
| QLoRA | [`PEFT_QLORA/`](PEFT_QLORA/README.md) | `bert-base-uncased` | LoRA adapters and task head on a 4-bit quantized base model | Low-bit base weights plus low-rank updates |

Each experiment reports **accuracy** and **weighted F1** on the IMDb test split before and after fine-tuning. The repository includes bar charts for the before/after metric comparison and line-chart visualizations of baseline versus final accuracy. See the individual READMEs for exact configuration and output paths.

## Repository layout

```text
FINE_TUNING/
├── README.md
├── LICENSE
├── NEW_HEAD/
│   ├── README.md
│   ├── fine_tuning_new_head_addiiton.py
│   ├── BAR_CHART.png
│   └── LINE_CHART.png
├── LAST_LAYER/
│   ├── README.md
│   ├── last_layer_fine_tuning.py
│   ├── bar_chart.png
│   └── Line_chart.png
├── PEFT_LORA/
│   ├── README.md
│   ├── peft_lora.py
│   ├── bar_chart.png
│   └── line_chart.png
└── PEFT_QLORA/
    ├── README.md
    ├── peft_qlora.py
    ├── bar_chart.png
    └── line_chart.png
```

The filenames and capitalization above reflect the current repository. GitHub paths are case-sensitive in many environments, so preserve capitalization when referencing the chart files.

## What the experiments compare

### 1. Head-only fine-tuning

[`NEW_HEAD`](NEW_HEAD/README.md) loads DistilBERT with a binary sequence-classification head, freezes the model's `base_model` parameters, and updates only the classification head. This is the least invasive of the four experiments and provides a useful reference for how much an initialized task head can learn while the pretrained representation remains fixed.

### 2. Last-layer fine-tuning

[`LAST_LAYER`](LAST_LAYER/README.md) freezes model parameters first, then enables gradients for the classification head and the final Transformer block. The implementation includes model-family checks for DistilBERT, BERT, RoBERTa/XLM-R, and a fallback for common `base_model` encoder layouts. The configured run uses DistilBERT and a lower learning rate than the head-only experiment.

### 3. LoRA with PEFT

[`PEFT_LORA`](PEFT_LORA/README.md) applies Low-Rank Adaptation to the query and value projection modules of `bert-base-uncased`. The base sequence-classification model is wrapped with PEFT using `TaskType.SEQ_CLS`; the adapter rank, scaling factor, and dropout are configured explicitly. The script prints the trainable-parameter summary before training.

### 4. QLoRA with PEFT and 4-bit loading

[`PEFT_QLORA`](PEFT_QLORA/README.md) loads `bert-base-uncased` through `BitsAndBytesConfig` with 4-bit NF4 quantization and double quantization, then applies the same family of LoRA query/value adapters. The configured compute dtype is BF16 when CUDA is available and FP32 otherwise. Its training arguments also enable BF16 when CUDA is available and disable FP16.

## Common training and evaluation protocol

The scripts share a recognizable sequence:

1. Set a reproducible seed with `set_seed(42)` and disable Weights & Biases reporting through `WANDB_DISABLED` / `report_to="none"`.
2. Load the IMDb dataset with `load_dataset("imdb")`.
3. Use the configured `SMALL_RUN` switch to sample 4,000 training examples and 2,000 test examples after seeded shuffling. In the scripts, `SMALL_RUN = True` by default to make demonstrations faster.
4. Load a fast tokenizer and tokenize the review `text` field with truncation enabled.
5. Use `DataCollatorWithPadding` to dynamically pad batches where included in the script.
6. Load a two-label sequence-classification model and configure which parameters are trainable for the method being demonstrated.
7. Evaluate the untrained task head/adapters or the current model before training.
8. Train for three epochs using the method-specific learning rate, batch sizes, precision settings, and weight decay.
9. Evaluate again, plot the before/after scores, and save the fine-tuned model and tokenizer under the corresponding `outputs/` directory.

The common metric function uses `argmax` over logits for predicted labels, computes accuracy, and computes **weighted** F1. The binary label convention in the inference helpers is `0 = negative` and `1 = positive`, displayed as `neg` and `pos` respectively.

## Results and chart interpretation

The charts stored in the repository are intended to compare metrics before and after fine-tuning for each strategy:

| Experiment | Accuracy/F1 plot | Accuracy comparison plot |
|---|---|---|
| Head-only | [Bar chart](NEW_HEAD/BAR_CHART.png) | [Line chart](NEW_HEAD/LINE_CHART.png) |
| Last-layer | [Bar chart](LAST_LAYER/bar_chart.png) | [Line chart](LAST_LAYER/Line_chart.png) |
| LoRA | [Bar chart](PEFT_LORA/bar_chart.png) | [Line chart](PEFT_LORA/line_chart.png) |
| QLoRA | [Bar chart](PEFT_QLORA/bar_chart.png) | [Line chart](PEFT_QLORA/line_chart.png) |

**Important plotting detail:** the current scripts construct the line plots by repeating the single baseline accuracy and the single final accuracy across an epoch-indexed x-axis. They are before-versus-after summary visualizations, not measurements logged separately at each epoch. The bar charts compare the baseline evaluation values with the post-training evaluation values. Exact scores should be read from the current chart artifacts or the corresponding run's evaluation output; this README does not invent numeric scores that are not documented in the source.

The original project provides a qualitative efficiency comparison: head-only updates roughly 1–2% of parameters, last-layer plus head roughly 10–15%, and LoRA/QLoRA roughly 1%; it labels the memory use as low, medium, low, and ultra-low respectively, with QLoRA using 4-bit quantization. These are the repository's stated high-level comparisons, not hardware-independent guarantees. Actual memory usage and the trainable-parameter percentage depend on the model architecture, library versions, precision, and runtime.

## Environment and installation

The scripts use the Hugging Face model and dataset download APIs, so the runtime needs internet access on the first run unless model and dataset artifacts are already cached. A CUDA GPU is helpful for speed and is needed for the intended 4-bit bitsandbytes experiment in most supported setups.

Install the shared libraries in a notebook cell or terminal environment:

```bash
pip install -U transformers datasets evaluate accelerate peft bitsandbytes matplotlib torch
```

For head-only and last-layer experiments, the project READMEs use the smaller set `transformers datasets evaluate accelerate`; LoRA and QLoRA also use `peft` and `bitsandbytes`. The QLoRA source notebook explicitly upgrades `bitsandbytes`, `transformers`, `accelerate`, and `peft`.

### Running the exported scripts

The `.py` files in this repository are exported from Colab notebooks. Their first lines include notebook-style `!pip` installation commands, which are valid in notebook cells but are not ordinary Python syntax in a plain `.py` process. For the least-friction reproduction path, open the corresponding notebook environment and run the cells as intended; alternatively, install dependencies separately in the shell and remove the notebook-only installation lines before running the exported file as a standalone Python script.

Typical experiment outputs are written to:

- `outputs/imdb_head_only/model/`
- `outputs/imdb_last_layer/model/`
- `outputs/imdb_lora/model/`
- `outputs/imdb_qlora/model/`

These paths are created by the scripts during execution and are not part of the tracked repository tree shown here. `trainer.save_model(...)` saves the fine-tuned model and `tokenizer.save_pretrained(...)` saves tokenizer artifacts beside it. Keep the resulting model directory if you want to reuse the trained artifact for inference.

## Inference concept

The head-only, last-layer, and LoRA scripts define a `predict(texts)` helper. It tokenizes a list of reviews with truncation and padding, moves the encoded batch to the configured device, switches the model to evaluation mode, disables gradient calculation, applies softmax to logits, and returns each input with negative probability, positive probability, and the predicted label. The READMEs include the same two sample review themes: one strongly positive review and one negative review. Those outputs are illustrative examples, not guaranteed probabilities from every newly trained run.

The QLoRA training script saves the adapted model and tokenizer; it does not currently include the same explicit `predict(texts)` helper in the exported code. The shared high-level goal is still binary IMDb sentiment classification.

## Reproducibility and customization

- **Fast demonstration:** keep `SMALL_RUN = True`; seeded subsets use 4,000 train and 2,000 test examples.
- **Full IMDb split:** set `SMALL_RUN = False` in the script to avoid the subset selection. Training and evaluation will take longer and require more compute.
- **Change the pretrained checkpoint:** edit `model_name` and verify that the last-layer selector or LoRA target-module names match the chosen architecture.
- **Inspect the trainable surface:** the scripts print trainable parameter names or PEFT trainable-parameter summaries for sanity checking.
- **Compare runs consistently:** record model checkpoint, subset mode, library versions, GPU model, precision, and the final `eval_accuracy` / `eval_f1` values alongside the charts.

## Scope and interpretation

This is a collection of fine-tuning experiments and result visualizations, not a packaged deployment service. The source demonstrates the training/evaluation workflow and saving of model artifacts. It does not define a shared CLI, inference API, hyperparameter sweep, or model-serving endpoint. Results from the small sampled run should be interpreted as a fast demonstration on a subset rather than a full-dataset benchmark. The PEFT and quantization approaches are intended to reduce training cost relative to updating a full backbone; they do not guarantee the same absolute quality or memory footprint on every model or hardware configuration.

## Author

Shubham Singh


---

## License

This project is licensed under the [MIT License](LICENSE). The full license text is reproduced below; the standalone [`LICENSE`](LICENSE) file is the canonical license file for this directory.

MIT License

Copyright (c) 2026 Shubham Kumar Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
