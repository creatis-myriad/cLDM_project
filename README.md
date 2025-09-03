# Controllable latent diffusion model to evaluate the performance of cardiac segmentation methods



## Prerequisites
- Python 3.10+


## Installation
- Clone the repository and enter it:
    - git clone <https://github.com/creatis-myriad/cLDM_project>
    - cd <cLDM_poject/bin/>


## Create the Conda environment
- Create from the provided environment file:
    - conda env create -f environment.yml
- Activate it:
    - conda activate cLDM_env


## Data preparation
- Data from MYOSAIQ challenge were put in one folder and divided by patients.
- The D8 subset was not used during training.
- In the code, `D_metrics` is a dictionnary with specific informations (like the metric) obtained from the MYOSAIQ database.


## Configuration
TODO
<!-- - Configuration files are typically under configs/ (e.g., configs/train.yaml, configs/infer.yaml).
- Common keys:
    - data: paths to training/validation/test sets
    - model: architecture and checkpoint settings
    - train: batch size, epochs, optimizer, scheduler
    - infer: sampling steps, guidance scales, output dirs
- Override any setting via CLI flags if supported:
    - python scripts/train.py --config configs/train.yaml train.batch_size=8 data.root=./data -->


## How to run
TODO
<!-- - Training:
    - python scripts/train.py --config configs/train.yaml
- Resume training:
    - python scripts/train.py --config configs/train.yaml train.resume_from=path/to/checkpoint.ckpt
- Inference/sampling:
    - python scripts/infer.py --config configs/infer.yaml infer.ckpt=path/to/checkpoint.ckpt infer.output_dir=outputs/
- Evaluation (e.g., segmentation metrics):
    - python scripts/eval.py --config configs/eval.yaml eval.pred_dir=outputs/ eval.gt_dir=data/gt/
- Replace script paths/names with those in the repository. All parameters can be adjusted in the YAML or overridden on the CLI. -->


## Figure
TODO
<!-- 
File fig_ ... was the one use to create fig 2 and 3 in the article.
 -->












