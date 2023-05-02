# Matrix Compressor

This project is an implementation of the LPLR framework for applying a joint low rank quantization technique to compress matrices, with applications in domains with large data matrices. Popular applications include Deep Learning, Graph (Neural) Networks, Computational Biology, Large scale data retrieval amongst others.

## Setup

The repository uses `conda` for setup, other python dependency management frameworks may be utilized as appropriate by importing `environment.yml`.

```bash
conda import -f environment.yml
```

## Common operations

### LlaMa Layer Analysis

Replace the MODEL_DIRECTORY by the location of the unquantized model, and adjust parameters `b1`, `b2` and `cr`.

```bash
export OUTPUT_DIRECTORY="./artifacts/llama-quantized"
export MODEL_DIRECTORY="./artifacts/llama-7b-hf/"
export LOGURU_LEVEL=INFO 
stdbuf -oL python scripts/llama/layer_wise_quantization.py --model-directory $MODEL_DIRECTORY --output-directory $OUTPUT_DIRECTORY --b1 8 --b2 8 --cr 1 --map-location "cuda:1" 2>&1 | stdbuf -oL tee -i $OUTPUT_DIRECTORY/quantization-$(date +%m%d%H%M%S).log
```

### Quantization and Evaluation

```bash
export RF=0.8
export B1=16
export B2=16
export CUDA_VISIBLE_DEVICES=1
export LOGURU_LEVEL=TRACE
export INPUT_DIR=artifacts/llama-7b-hf
export OUTPUT_DIR=artifacts/llama-quantized-svd-r{$RF}-{$B1}-{$B2}

stdbuf --output=L python scripts/lplr/quantize_torch_splits_svd.py --in-path $INPUT_DIR --out-path $OUTPUT_DIR --map-device 'cuda:0' --rank-fraction $RF --b1 $B1 --b2 $B2 2>&1 | stdbuf --output=L tee $OUTPUT_DIR/quantization.log

cp artifacts/llama-7b-hf/tokenizer.model artifacts/llama-7b-hf/*.json $INPUT_DIR

stdbuf --output=L python repos/lm-evaluation-harness/main.py --model hf-causal --model_args pretrained=/home/rsaha/varun/matrix-compressor/$INPUT_DIR --tasks boolq,hellaswag,piqa 2>&1 | stdbuf --output=L tee $INPUT_DIR/evaluation.log
```