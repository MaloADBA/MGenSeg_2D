# Cross-Modality segmentation : M-GenSeg 2D (https://arxiv.org/abs/2212.07276)

## Initialization

Clone repo "git clone https://github.com/MaloADBA/MGenSeg_2D.git"

Run "pip install -r requirements.txt" to install dependencies. 

Run "git submodule init" to initialize submodules.

Run "git submodule update" to download submodules.

## This repo handles several domain adaptation models for cross-modality segmentation :

'M-GenSeg' (https://arxiv.org/abs/2212.07276)

'AccSegNet' (https://link.springer.com/chapter/10.1007/978-3-030-87193-2_5)

'AttENT' (https://ieeexplore.ieee.org/document/9669620)

'UAGAN' (https://arxiv.org/abs/1907.03548)

'Supervised TransUnet' (https://arxiv.org/abs/2102.04306) --> For comparison with a fully supervised model

## Task : Cross-modality tumor segmentation with BRATS

We build an unsupervised domain adaptation task with BraTS where each MR contrast (T1,T1ce,FLAIR,T2) is considered as a distinct modality. The models provided aim at reaching good segmentation performances on an unlabeled target modality dataset by leveraging annotated source images of an other modality.

We use the 2020 version of the BRATS data from https://www.med.upenn.edu/cbica/brats2020/data.html. Once downloaded to `<download_dir>`, this data can be prepared for domain adaptation tasks using a provided script, as follows:

```
python scripts/data_preparation/Prepare_multimodal_brats_2D.py --data_dir "<download_dir>" --save_to "/path/data.h5" --min_tumor_fraction 0.01 --min_brain_fraction 0.25 --no_crop --skip_bias_correction

```
Data preparation creates a new dataset based on BRATS that contains 2D hemispheres, split into sick and healthy subsets for each possible contrast (T1,T1ce,FLAIR,T2).

### Launching experiments

#### Launching training for MGenSeg

In this example, the following model configuration is used:
`model/configs/brats_2017/bds3/bds3_003_xid50_cyc50.py`

In this specific configuration file, the decoder is mostly shared for the segmentation path. When run in `mode=0` (passed in `forward` call; default), the decoder outputs an image, with `tanh` normalization at the output; when run in `mode=1`, it outputs a segmentation mask, with `sigmoid` normalization at the output. Modes 0 and 1 differ in three ways:
1. The final norm/nonlinearity/convolution block is unique for each mode.
2. The final nonlinearity is `tanh` in mode 0 and `sigmoid` in mode 1.
3. Every block in the decoder is normalized with `layer_normalization` in mode 0 and with adaptive instance normalization in mode 1.
Adaptive instance normalization uses parameters predicted by an MLP when the decoder is run in mode 0. They are passed to the mode 1 decoder as `skip_info`.

An example of an experiment launched with this config is:
```
python brats_segmentation.py --path "experiments/brats_2017/bds3/bds3_003_xid50_cyc50 (f0.01, D_lr 0.001) [b0.3]" --model_from model/configs/brats_2017/bds3/bds3_003_xid50_cyc50.py --batch_size_train 20 --batch_size_valid 20 --epochs 1000000 --rseed 1234 --optimizer '{"G": "amsgrad", "D": "amsgrad"}' --opt_kwargs '{"G": {"betas": [0.5, 0.999], "lr": 0.001}, "D": {"betas": [0.5, 0.999], "lr": 0.01}}' --n_vis 8 --weight_decay 0.0001 --dataset brats17 --orientation 1 --data_dir=./data/brats/2017/hemispheres_b0.3_t0.01/ --labeled_fraction 0.01 --augment_data --nb_proc_workers 2
```

Optimizer arguments are passed as a JSON string through the `--opt_kwargs` argument.

Note that if `CUDA_VISIBLE_DEVICES` is not set to specify which GPUs to use, the model will attempt to use all available GPUs. The code is multi-GPU capable but no serious training has been done on multiple GPUs. Use multiple GPUs with caution. There have been bugs in pytorch (hopefully fixed now) that made it either cause some layers to fail to be updated or fail to be resumed.

#### Example: resuming a brats experiment

Resuming the above experiment could be done with:
```
python brats_segmentation.py --path "experiments/brats_2017/bds3/bds3_003_xid50_cyc50 (f0.01, D_lr 0.001) [b0.3]"
```

Upon resuming, the model configuration file is loaded from the saved checkpoint. All arguments passed upon initializing the experiment are loaded as well. **Any of these can be over-ridden by simply passing them again with the resuming command.**

## Dispatching on a Compute Canada

Experiments should be launched from one of the login nodes of a compute canada cluster. The launcher then sets up and queues the job on the cluster.

Note: SLURM setup for compute canada could be easily extended to other SLURM based clusters.

To the task arguments, add the argument `--dispatch_dgx`, along with any aditional DGX-specific arguments:

`--account` : the compute canada account to use for requesting resources  
`--cca_gpu` : number of GPUs to request  
`--cca_cpu` : number of CPU cores to request  
`--cca_mem` : amount of memory to request, as a string (eg. '12G')  
`--time` : the amount of time to request the job for (see `sbatch` time syntax)  

When dispatching on a compute canada cluster, a daemon is created that requeues any jobs that time out, allowing them to resume. This allows requesting a short run time which makes it much more likely to get high priority resources; an optimal run time request is for 3h (`--time "3:0:0"`).
