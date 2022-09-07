# Pyramid Vision Transformer <a href="https://arxiv.org/abs/2106.13797"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a></h1> 
This repo contains the **unofficial** JAX/Flax implementation of <a href="https://arxiv.org/abs/2106.13797">PVT v2: Improved Baselines with Pyramid Vision Transformer</a>. <br/>
All credits to the authors **Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao** for their wonderful work.

## Datasets
- **TensorFlow Datasets** - Refer to [TensorFlow Dataset Image Classification Catalog](https://www.tensorflow.org/datasets/catalog/overview#image_classification) and accordingly modify the following keys in `config/default.py`.
```ruby
config.dataset_name = "mnist"   # name should be present in tensorflow catalog
config.data_shape = [28, 28, 1]   # shape of images in the dataset
config.num_classes = 10   # numbler of classes in the dataset
```
- **PyTorch DataLoader** - To load datasets in PyTorch style, use the wrapper for torch.DataLoader in `data/numpyloader.py` -> `NumpyLoader` along with a custom collate function.
- **Custom Dataset** - Currently, this repo does not support out of the box support for custom image classification dataset. However, you can manipulate `NumpyLoader` to accomplish this.

## Dependencies
*It is recommended to create a new virtual environment so that updates/downgrades of packages do not break other projects.*<br/>
- Environment characteristics:<br/>`python = 3.9.12` `cuda = 11.3` `jax = 0.3.16` `flax = 0.6.0`
```
pip install -r requirements.txt
```
*Note: Flax is not dependent on TensorFlow itself, however, we make use of methods that take advantage of `tf.io.gfile`* As such, we only install `tensorflow-cpu`. Same is the case with PyTorch, we only install it in order to use their `torch.data.DataLoader`.

## Run
To get started, clone this repo and install the dependencies.<br/>
### Training
- Configure the **{key: value pairs}** in the config file present at `config/default.py`.<br/>
- Execute train.py with path to checkpoint and --eval-only argument. Example usage:
```python
python run.py --model-name "PVT_V2_B0" --work-dir "output/"
```

### Evaluation
- Execute train.py with appropriate arguments. Example usage:
```python
python run.py --model-name "PVT_V2_B0" \
              --eval-only \
              --checkpoint_dir "output/"
```

## Acknowledgements
We acknowledge the excellent implementation of PVT in [MMDetection](https://github.com/open-mmlab/mmdetection), [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) and the [official implementation](https://github.com/whai362/PVT). I repeatedly referred to these implementations as a reference.

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
