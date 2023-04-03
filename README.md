# SOLIDER on [Pedestrian Detection]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-appearance-a-semantic-controllable/pedestrian-detection-on-citypersons)](https://paperswithcode.com/sota/pedestrian-detection-on-citypersons?p=beyond-appearance-a-semantic-controllable)

This repo provides details about how to use [SOLIDER](https://github.com/tinyvision/SOLIDER) pretrained representation on pedestrian detection task.
We modify the code from [Pedestron](https://github.com/hasanirtiza/Pedestron), and you can refer to the original repo for more details.

## Installation and Datasets

Details of installation and dataset preparation can be found in [Pedestron](https://github.com/hasanirtiza/Pedestron).

## Prepare Pre-trained Models 
You can download models from [SOLIDER](https://github.com/tinyvision/SOLIDER), or use [SOLIDER](https://github.com/tinyvision/SOLIDER) to train your own models.
Before training, you should use `convert_model_version.py` to convert the pretrained models to the right version.

## Training

Train with single GPU:

```shell
python tools/train.py ${CONFIG_FILE}
```

Train with multiple GPUs:
```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

`CONFIG_FILE` can be found in `configs/solider`.

Test:

```shell
sh run.sh
```

## Performance

| Method | Model | CityPersons<br>(MR-2) |
| ------ | :---: | :---: |
| SOLIDER | Swin Tiny | 10.3/40.8 |
| SOLIDER | Swin Small | 10.0/39.2 |
| SOLIDER | Swin Base | 9.7/39.4 |

- We use the pretrained models from [SOLIDER](https://github.com/tinyvision/SOLIDER).
- The semantic weight is set to 1.0 in these experiments.

## Citation

If you find this code useful for your research, please cite our paper

```
@inproceedings{chen2023beyond,
  title={Beyond Appearance: a Semantic Controllable Self-Supervised Learning Framework for Human-Centric Visual Tasks},
  author={Weihua Chen and Xianzhe Xu and Jian Jia and Hao Luo and Yaohua Wang and Fan Wang and Rong Jin and Xiuyu Sun},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```
