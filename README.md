# Pyramid Vision Transformer
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/JAX-0.3.16-orange.svg)](https://github.com/google/jax)
<a href="https://arxiv.org/abs/2106.13797"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a></h1> 

This repo contains the **unofficial** JAX/Flax implementation of <a href="https://arxiv.org/abs/2106.13797">PVT v2: Improved Baselines with Pyramid Vision Transformer</a>. <br/>
All credits to the authors **Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao** for their wonderful work.

## Dependencies
*It is recommended to create a new virtual environment so that updates/downgrades of packages do not break other projects.*<br/>
- Environment characteristics:<br/>`python = 3.9.12` `cuda = 11.3` `jax = 0.3.16` `flax = 0.6.0`
- Follow the instructions on [official JAX/Flax documentation](https://flax.readthedocs.io/en/latest/installation.html) for installing their packages.

  ```
  pip install -r requirements.txt
  ```

*Note: Flax is not dependent on TensorFlow itself, however, we make use of methods that take advantage of `tf.io.gfile`* As such, we only install `tensorflow-cpu`. Same is the case with PyTorch, we only install it in order to use their `torch.data.DataLoader`.

## Run
To get started, clone this repo and install the required dependencies.<br/>

### Datasets
- **TensorFlow Datasets** - Refer to [TensorFlow Dataset Image Classification Catalog](https://www.tensorflow.org/datasets/catalog/overview#image_classification) and accordingly modify the following keys in `config/default.py`.

  ```python
  config.dataset_name = "imagenette"
  config.data_shape = [224, 224]
  config.num_classes = 10
  config.split_keys = ["train", "validation"]
  ```

- **PyTorch DataLoader** - To load datasets in PyTorch style, use the wrapper for torch.DataLoader in `data/numpyloader.py` -> `NumpyLoader` along with a custom collate function.
- **Custom Dataset** - Currently, this repo does not support out of the box support for custom image classification dataset. However, you can manipulate `NumpyLoader` to accomplish this.

### Training
- Configure the **{key: value pairs}** in the config file present at `config/default.py`.<br/>
- Execute train.py with path to checkpoint and --eval-only argument. Example usage:

  ```python
  python train.py --model-name "PVT_V2_B0" --work-dir "output/"
  ```

### Evaluation
- Execute train.py with appropriate arguments. Example usage:

  ```python
  python train.py --model-name "PVT_V2_B0" \
                  --eval-only \
                  --checkpoint_dir "output/"
  ```
## To do
- [ ] Convert ImageNet pretrained PyTorch weights (.pth) to Flax weights

*Note: Since my undergrad studies are resuming after summer break, I may or may not be able to find time to complete the above tasks. 
If you want to implement the aforelisted tasks, I'll be more than glad to merge your pull request. ❤️*

## Acknowledgements
We acknowledge the excellent implementation of PVT in [MMDetection](https://github.com/open-mmlab/mmdetection), [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) and the [official implementation](https://github.com/whai362/PVT). I referred to these implementations as a source of reference.

## Citing PVT
- **PVT v1**

  ```
  @inproceedings{wang2021pyramid,
    title={Pyramid vision transformer: A versatile backbone for dense prediction without convolutions},
    author={Wang, Wenhai and Xie, Enze and Li, Xiang and Fan, Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, Ping and Shao, Ling},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={568--578},
    year={2021}
  }
  ```

- **PVT v2**

  ```
  @article{wang2021pvtv2,
    title={Pvtv2: Improved baselines with pyramid vision transformer},
    author={Wang, Wenhai and Xie, Enze and Li, Xiang and Fan, Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, Ping and Shao, Ling},
    journal={Computational Visual Media},
    volume={8},
    number={3},
    pages={1--10},
    year={2022},
    publisher={Springer}
  }
  ```
