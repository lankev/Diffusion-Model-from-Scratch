---



\# Diffusion Model from Scratch



\*\*DDPM + DDIM + Conditional Generation + Modular QBlock Architecture\*\*



\## Overview



This repository contains a complete implementation of a diffusion model built from scratch in PyTorch.

The project includes:



\* Denoising Diffusion Probabilistic Model (DDPM)

\* Denoising Diffusion Implicit Model (DDIM)

\* Conditional image generation (class-conditioned)

\* Modular U-Net architecture

\* Optional QBlock extension (quantum-ready architectural module)

\* Experimental benchmarking and evaluation pipeline

\* Robust checkpoint handling and architecture detection



The objective of this project is to provide a clear, modular, and extensible implementation of diffusion models, suitable for educational, research, and experimentation purposes.



---



\## Technical Scope



The implementation covers:



\* Forward diffusion process

\* Noise prediction training objective

\* Reverse denoising process

\* DDPM probabilistic sampling

\* DDIM deterministic accelerated sampling

\* Conditional embeddings (time + class)

\* Modular architectural extensions

\* Benchmarking and performance evaluation



The codebase avoids external diffusion libraries to ensure full control and understanding of the model internals.



---



\## Model Architecture



\### Conditional U-Net



The core architecture is a conditional U-Net composed of:



\* Sinusoidal time embeddings

\* Learnable class embeddings

\* Residual convolutional blocks

\* Downsampling and upsampling layers

\* Optional QBlock module



Time and class embeddings are injected throughout the network to enable conditional generation.



\### QBlock (Optional)



The QBlock is a modular architectural extension designed to:



\* Be enabled or disabled via configuration

\* Remain checkpoint-compatible

\* Support future hybrid classical–quantum experimentation



The evaluation scripts automatically detect whether a checkpoint contains QBlock weights and instantiate the correct architecture accordingly.



---



\## DDPM vs DDIM



\### DDPM



\* Probabilistic reverse diffusion process

\* High stability

\* Typically requires ~1000 denoising steps

\* Higher inference time



\### DDIM



\* Deterministic sampling

\* Significantly fewer steps (10–200)

\* Substantial inference speed improvements

\* Uses the same trained model



\### Experimental Results



| Method | Steps | Approx. Time |

| ------ | ----- | ------------ |

| DDPM   | 1000  | ~38 seconds  |

| DDIM   | 50    | ~1.7 seconds |

| DDIM   | 10    | ~0.3 seconds |



The results demonstrate that DDIM provides up to 100x speedup with limited quality degradation.



---



\## Repository Structure



```

r3\_lib.py                         Core diffusion implementation and architecture

r3\_train\_conditional.py          Conditional training script

r3\_sample\_conditional\_grid.py    Conditional sampling

r3\_sample\_ddim\_speed\_compare.py  Speed benchmarking

r3\_viz\_denoise\_steps.py          Denoising visualization

r4\_compare\_ddpm\_ddim\_final.py    Final benchmark evaluation

r4\_metrics\_samples.py            Statistical evaluation

r4\_plot\_loss.py                  Training loss visualization

r4\_ablation\_qblock.py            Optional QBlock ablation

run\_menu.ps1                     Interactive launcher (Windows)

release4/                        Generated evaluation results

```



---



\## Installation



\### Requirements



\* Python 3.10+

\* PyTorch

\* torchvision

\* matplotlib

\* tqdm



\### Setup



Create a virtual environment:



```bash

python -m venv .venv

```



Activate environment (Windows PowerShell):



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```bash

pip install torch torchvision matplotlib tqdm

```



---



\## Usage



\### Train Conditional Model



```bash

python r3\_train\_conditional.py

```



\### Generate Conditional Samples



```bash

python r3\_sample\_conditional\_grid.py

```



\### Benchmark DDPM vs DDIM



```bash

python r4\_compare\_ddpm\_ddim\_final.py

```



\### Compute Statistical Metrics



```bash

python r4\_metrics\_samples.py

```



\### Plot Training Loss



```bash

python r4\_plot\_loss.py

```



\### Interactive Execution (Windows)



```powershell

.\\run\_menu.ps1

```



---



\## Evaluation Pipeline



The project includes:



\* Training loss curve visualization

\* DDPM vs DDIM runtime benchmarking

\* Pixel-wise statistical evaluation:



&nbsp; \* Mean

&nbsp; \* Standard deviation

&nbsp; \* Minimum / Maximum values

\* Automatic QBlock detection

\* Robust checkpoint loading

\* Atomic CSV writing for Windows compatibility



The evaluation confirms:



\* Stable convergence

\* Consistent conditional generation

\* Significant inference acceleration with DDIM



---



\## Design Principles



\* Modularity

\* Explicit control over diffusion mathematics

\* Architecture–checkpoint consistency

\* Robust error handling

\* Clear separation between training and evaluation



---



\## Future Work



\* Integration of true quantum circuits

\* FID or Inception Score evaluation

\* Scaling to larger datasets

\* Latent diffusion models

\* Hybrid classical–quantum experimentation



---



\## License



MIT License



---



\## Author



Developed as part of an advanced deep learning implementation project focused on generative modeling and diffusion architectures.



