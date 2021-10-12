# Climate Modeling with Neural Diffusion Equation

## Dependencies
```{bash}
conda create -n nde python==3.8.0
conda install pytorch==1.7.0 cudatoolkit=11.0 -c pytorch
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-geometric
pip install pyyaml
pip install tensorboardX
pip install torchdiffeq
```

or you can install conda environment via `environment.yml`

```{bash}
conda env create -f environment.yml
```

## How to run

One-step prediction for LA Dataset
```{bash}
bash run.sh
```