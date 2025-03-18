# DSTAdapter:Divided Spatial-Temporal Adapter Fine-tuning Method for Sign Language Recognition

This repository contains the model files and parameter files for DSTAdapter, a parameter efficient fine-tuning isolate sign language recognition model based on MMAction2 framework. The model achieves competitive performance on four datasets: INCLUDE, AUTSL, Bukva, and LSA64.

## Model Performance

The performance of the model on the four datasets is as follows:

| Dataset       | Model Size | Top-1 Accuracy | Top-5 Accuracy | Notes          |
|---------------|----------------|----------------|----------------|----------------|
| INCLUDE   | Vit-B <br> Vit-L | 95.29% <br> 97.06% | 99.41% <br> 99.12% | trained for 100 epochs |
| AUTSL   | Vit-B <br> Vit-L | 93.87% <br> 94.91% | 98.85% <br> 99.41% | trained for 100 epochs |
| Bukva   | Vit-B <br> Vit-L | 95.44% <br> 96.62% | 99.26% <br> 99.17% | trained for 100 epochs |
| LSA64   | Vit-B <br> Vit-L | 100% <br> 100% | 100% <br> 100% | trained for 50 epochs |

*Note: The performance metrics are based on the provided model and parameter files.*

---

## Getting Started

### Prerequisites

To use this model, you need to have the following installed:

- Python 3.7 or higher
- PyTorch 1.8 or higher
- [MMAction2](https://github.com/open-mmlab/mmaction2) framework

### Installation

1. Clone the MMAction2 repository:
   ```bash
   git clone https://github.com/open-mmlab/mmaction2.git
   cd mmaction2
   
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Install MMAction2:
   ```bash
   pip install -v -e .

### Model and Parameter Files

This repository provides the following files:

`mmaction2/models/backbones/dstadapter.py`: The model architecture file.
`mmaction2/configs/recognition/dst_adapter/stadapter_vit_[base/large]_[dataset_name].py`: The parameter file.

### Usage

1. Place the `model file` and `parameter file` files in the appropriate directory within the MMAction2 framework.
2. Modify the MMAction2 configuration file to load the model and parameters. For example:
```python

model = dict(
    type='YourModelType',
    backbone=dict(
        type='YourBackboneType',
        ...
    ),
    ...
)
```


3. Run the inference or training script provided by MMAction2. For example:
```bash
python tools/train.py path/to/config.py path/to/parameter.py
```

## Dataset Information

The model was evaluated on the following datasets:

1. INCLUDE: India sign language dataset. [download](https://zenodo.org/records/4010759)
2. AUTSL: Turkish sign language dataset. [download](https://cvml.ankara.edu.tr/datasets/)
3. Bukva: Russia sign language dataset. [download](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/bukva/bukva.zip)
4. LSA64:  Argentine sign language dataset. [download](https://facundoq.github.io/datasets/lsa64/)

Note: The datasets are not included in this repository. Please refer to the official dataset websites for downloading and setup instructions.

## Acknowledgments

1. Thanks to the MMAction2 team for providing the framework.
2. This work was supported by [Your Funding Source, if applicable].
3. Thanks to the dataset provider that we used in this work. 

