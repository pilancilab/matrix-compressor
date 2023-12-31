# Matrix Compressor

This repository contains code for the paper [Matrix Compression via Randomized Low Rank and Low Precision Factorization](https://arxiv.org/abs/2310.11028) by **Rajarshi Saha**, **Varun Srivastava** and **Mert Pilanci (NeurIPS 2023)**.

We propose **LPLR** - a framework for joint rank reduction and quantization to compress large data matrices, with applications in domains such as Natural Language Processing (LLMs), Big Data Processing (retrieval and compression) and more.

LPLR exploits low rank structure to obtain a low rank decomposition of any matrix $\mathbf{A}$ as $\mathbf{A} \approx \mathbf{L}\mathbf{R}$, where $\mathbf{L}$ and $\mathbf{R}$ are the low rank factors.
The total number of elements in $\mathbf{L}$ and $\mathbf{R}$ can be significantly less than that in $\mathbf{A}$.
Furthermore, the entries of $\mathbf{L}$ and $\mathbf{R}$ are quantized to low precision formats -- compressing $\mathbf{A}$ by giving us a low rank and low precision factorization.
Our algorithm first computes an approximate basis of the range space of $\mathbf{A}$ by randomly sketching its columns, followed by a quantization of the vectors constituting this basis.
It then computes approximate projections of the columns of $\mathbf{A}$ onto this quantized basis.
The tradeoff between compression ratio and approximation accuracy allows for flexibility in choosing these parameters based on specific application requirements.

### Algorithm

**LPLR: Randomized Low-Precision Low-Rank factorization**

**Input:** Matrix $\mathbf{A} \in \mathbb{R}^{n \times d}$, sketch size $m$, Quantizers $\mathrm{Q}$, $\mathrm{Q}'$ with dynamic ranges $\mathrm{R}\_\mathrm{Q}$, $\mathrm{R}\_{\mathrm{Q}'}$ and bit-budgets $\mathrm{B}, \mathrm{B}'$ respectively.

**Output:** Factorization: $\mathbf{L}\mathbf{R}$ where $\mathbf{L} \in \mathbb{R}^{n \times m}$, $\mathbf{R} \in \mathbb{R}^{m \times d}$

- Sample a Gaussian sketching matrix $\mathbf{S} \in \mathbb{R}^{d \times m}$ with entries $S_{ij} \sim {\cal N}\left(0, \frac{1}{m}\right)$.
- Compute an approximate basis of column space of $\mathbf{A}$ by forming the sketch: $\mathbf{A}\mathbf{S}$.
- Quantize the approximate basis with $\mathrm{Q}$ to get $\mathrm{Q}(\mathbf{A}\mathbf{S})$.
- Find $\mathbf{W}^* = {\rm argmin}\_{\mathbf{W}}\left\lVert{\mathrm{Q}(\mathbf{A}\mathbf{S})\mathbf{W} - \mathbf{A}}\right\rVert\_{\rm F}^2$.
- Quantize $\mathbf{W}^\*$ using quantizer $\mathrm{Q}'$ to get $\mathrm{Q}'(\mathbf{W}^\*)$.

**Return**: Low-rank and low-precision approximation $\mathbf{L}\mathbf{R}$ where $\mathbf{L} = \mathrm{Q}(\mathbf{A}\mathbf{S})$, $\mathbf{R} = \mathrm{Q}'(\mathbf{W}^*)$.


## Citation
Please cite our work as:
```bibtex
@inproceedings{saha2023lplr,
    title={{Matrix Compression via Randomized Low Rank and Low Precision Factorization}},
    author={Rajarshi Saha, Varun Srivastava and Mert Pilanci},
    booktitle={Advances in Neural Information Processing Systems},
    year={2023}
}
```

## 🛠 Installation
Run the following command to install all dependencies in a conda environment named `lplr`. Change the cuda version for torch as required.
```
sh install.sh
```
After installation, activate the environment with
```
conda activate lplr
```

## 🚀 Trying out LPLR

You can try out the image compression experiments from the paper using `example.ipynb` notebook

<!-- ## 💡 Tips -->

## 📊 Replicating experiments in paper

### LlaMa Layer wise Analysis

Replace the MODEL_DIRECTORY by the location of the unquantized model, and adjust parameters `b1`, `b2` and `cr`.

```bash
export OUTPUT_DIRECTORY="./artifacts/llama-quantized"
export MODEL_DIRECTORY="./artifacts/llama-7b-hf/"
export LOGURU_LEVEL=INFO 
stdbuf -oL python scripts/llama/per_layer_naive_quantization_comparison/lplr_vanilla.py --model-directory $MODEL_DIRECTORY --output-directory $OUTPUT_DIRECTORY --b1 8 --b2 8 --cr 1 --map-location "cuda:1" 2>&1 | stdbuf -oL tee -i $OUTPUT_DIRECTORY/quantization-$(date +%m%d%H%M%S).log
```

Replace the file `lplr_vanilla.py` by the corresponding variant in the folder `scripts/llama/per_layer_naive_quantization_comparison` to use LSVD/DSVD.
