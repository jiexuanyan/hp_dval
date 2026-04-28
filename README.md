# Dual View Alignment Learning with Hierarchical Prompt for Class-Imbalance Multi-Label Image Classification[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/open-vocabulary-multi-label-classification/multi-label-zero-shot-learning-on-nus-wide)](https://paperswithcode.com/sota/multi-label-zero-shot-learning-on-nus-wide?p=open-vocabulary-multi-label-classification)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/open-vocabulary-multi-label-classification/multi-label-zero-shot-learning-on-open-images)](https://paperswithcode.com/sota/multi-label-zero-shot-learning-on-open-images?p=open-vocabulary-multi-label-classification)

This is the official PyTorch implementation for our paper **"Dual View Alignment Learning with Hierarchical Prompt for Class-Imbalance Multi-Label Image Classification" (HP-DVAL)**.

# Open-Vocabulary Multi-Label Classification via Multi-Modal Knowledge Transfer (AAAI 2023 Oral)

This repository provides code for both **Long-Tailed (LT)** and **Few-Shot Learning (FSL)** settings for multi-label image classification on common datasets.

![Framework](figures/mkt.jpg)

## Requirements

This is the official repository of our paper Open-Vocabulary Multi-Label Classification via Multi-modal Knowledge Transfer.

Please install the required packages first:

## Setup

```bash

pip install -r requirements.txt```bash

```pip install -r requirements.txt

```

## Datasets Preparation

## Preparation

Ensure you have downloaded the required datasets (e.g., COCO, NUS-WIDE, VOC) and placed them correctly in the `data/` and `idx/` directories before running the scripts.

1. Download pretrained VLP(ViT-B/16) model from [OpenAI CLIP](https://github.com/openai/CLIP).

---

2. Download images of NUS-WIDE dataset  from [NUS-WIDE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html).

## 1. Long-Tailed (LT) Tasks

3. Download annotations following the [BiAM](https://github.com/akshitac8/BiAM) from [here](https://drive.google.com/drive/folders/1jvJ0FnO_bs3HJeYrEJu7IcuilgBipasA?usp=sharing).

For the LT task, the training is divided into two stages:

- **First Stage**: Standard prompt learning or base training.4. Download other files from [here](https://drive.google.com/drive/folders/1kTb83_p92fM04OAkGyiHypOgwtxc4wVa?usp=sharing).

- **Second Stage**: Dual view alignment and hierarchical prompt tuning.

The organization of the dataset directory is shown as follows.

### Training on COCO-LT

```bash

**First Stage (CoOp / Base Model):**NUS-WIDE

```bash  ├── features

python coco_runner.py --dataset coco-lt --lr 1e-5 --loss_function bce --topk 32 --alpha 0.4  ├── Flickr

```  ├── Concepts81.txt

  ├── Concepts925.txt

**Second Stage (HP-DVAL):**  ├── img_names.pkl

```bash  ├── label_emb.pt

# After obtaining the first stage checkpoint, you can run the second stage.  └── test_img_names.pkl

# Please specify your first stage checkpoint path if necessary.```

python coco_runner_dual.py --dataset coco-lt --lr 1e-5 --loss_function bce --topk 32

```## Training MKT on NUS-WIDE



### Testing on COCO-LT```bash

python3 train_nus_first_stage.py \

To evaluate a trained checkpoint on the COCO-LT evaluation set:        --data-path path_to_dataset \

        --clip-path path_to_clip_model

```bash

python coco_test.py --resume path/to/your/checkpoint.ckpt --dataset coco-lt```

```

*(The script automatically adapts the model architecture based on the checkpoint's name, e.g., if it's a first stage or second stage checkpoint).*The checkpoint of the first training stage is [here](https://drive.google.com/file/d/158ntqLvepVklwmY1PvlqIhguv7wN6SZI/view?usp=sharing).



---```bash

python3 -m torch.distributed.launch --nproc_per_node=8 train_nus_second_stage.py \

## 2. Few-Shot Learning (FSL) Tasks        --data-path path_to_dataset \

        --clip-path path_to_clip_model \

For Few-Shot Learning tasks, the evaluation is based on K-shot examples (e.g., 5-shot).         --ckpt-path path_to_first_stage_ckpt

```

### Training for FSL (e.g., on Pascal VOC)

The checkpoint of the second training stage is [here](https://drive.google.com/file/d/1TBh1eWDLhHTjTfnRRfZULpe4DfPj7u9O/view?usp=sharing).

```bash

python fsl_runner.py --dataset voc --lr 1e-5 --topk 32 --alpha 0.4## Testing MKT on NUS-WIDE

```

*(You can switch `--dataset` to `coco` or `nus` depending on your target dataset).*```bash

python3 train_nus_second_stage.py --eval \

### Testing for FSL        --data-path path_to_dataset \

        --clip-path path_to_clip_model \

To evaluate the FSL performance given a trained checkpoint:        --ckpt-path path_to_first_stage_ckpt \

        --eval-ckpt path_to_first_second_ckpt \

```bash```

python fsl_eval.py --dataset voc --resume path/to/your/fsl_checkpoint.ckpt --shot 5 --lr 1e-5

```## Inference on A Single Image

*(Adjust the `--shot` parameter according to your evaluation protocol, such as 1 or 5.)*

```bash

---python3 inference.py \

        --data-path path_to_dataset \

## Checkpoints and Logs        --clip-path path_to_clip_model \

        --img-ckpt path_to_first_stage_ckpt \

- Output checkpoints are usually saved into the `MKT_LT_checkpoint/` directory.        --txt-ckpt path_to_second_stage_ckpt \

- Training logs are directed to directories such as `log/log_coco/` or `log/log_fsl/`.        --image-path figures/test.jpg

```

## Acknowledgement

## Acknowledgement

Parts of this code are adapted from open-source works including CLIP, BiAM, and timm. We thank the original authors for their contributions to the community.

We would like to thank [BiAM](https://github.com/akshitac8/BiAM) and [timm](https://github.com/rwightman/pytorch-image-models) for the codebase.

## License

MKT is MIT-licensed. The license applies to the pre-trained models as well.

## Citation

Consider cite MKT in your publications if it helps your research.

```bash
@article{he2022open,
  title={Open-Vocabulary Multi-Label Classification via Multi-modal Knowledge Transfer},
  author={He, Sunan and Guo, Taian and Dai, Tao and Qiao, Ruizhi and Ren, Bo and Xia, Shu-Tao},
  journal={arXiv preprint arXiv:2207.01887},
  year={2022}
}
```
