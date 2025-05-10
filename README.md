## Neuroinformatics: Neural Decoding Project
This project consists of two customized paper codebases and two stand-alone jupyter notebooks, corresponding to different methods developed to address the problem of decoding behaviour from neural recordings.

### Code Organization of Various Methods
**Note that this README is not on the main branch, but rather the pointcloud branch
| Report Section # (Method) | Location | Files Involved | Entry Point |
|---|---|---|---|
| Sec. 6.1 (Improved GMM) | main branch | whole repository, except for report pdf and `TemporalLFP_Transformer...ipynb` | `main.py` |
| Sec. 6.2 (LFP Movement Threshold) | pointcloud branch | `Transformer_Variable_Labelling.ipynb` | Jupyter Notebook |
| Sec. 6.3 (Point Transformer) | pointcloud branch | The `DAPT` submodule, `main_pointcloud.ipynb`, and `density_decoding/utils/data_utils.py` | `main_pointcloud.ipynb` |
| Sec. 6.4 (Statistical LFP Features) | main branch | `TemporalLFP_Transformer_FullPipeline(Schohastic_and_Statistics).ipynb` | Jupyter Notebook |

### Dependencies

Jupypter notebooks are self contained, and the dependencies for the `density_decoding` and `DAPT` codebases are listed below. They are tested on a Ubuntu 22.04 machine with x86_64 architecture.

#### `density_decoding` (GMM Baseline)
These instructions are adjusted to act as a foundation for setting up `DAPT`, for the original instructions given by the authors of `density_decoding`, please see main branch of this repository.
```
conda create -n neuraldecoding python=3.9
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install ONE-api ibllib ipykernel matplotlib scikit-learn pandas h5py iblatlas numpy\<2 git+https://github.com/cwindolf/isosplit@main
```

#### `DAPT` (Point Transformer)
The following instructions builds on top of `density_decoding`'s dependencies, so it must be set up first before running the following.

The step to install `Pointnet2_PyTorch` from source requires a GCC version less than 11.

```
pip install -r DAPT/requirements.txt    # (remove scipy's version specifications)
cd DAPT/extensions/chamfer_dist/ && python setup.py install && cd ../../
cd extensions/emd && python setup.py install && cd ../../../
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```