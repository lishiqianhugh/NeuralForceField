# Neural Force Field

Few-shot Learning of Generalized Physical Reasoning

<p align="left">
    <a href='https://NeuralForceField.github.io/'>
    <img src='https://img.shields.io/badge/Web-Page-yellow?style=plastic&logo=Google%20chrome&logoColor=yellow' alt='Web'>
    </a>
    <a href='https://drive.google.com/file/d/14IQ-DkVIQ0pvI9uXSgrc6j9Wjk1E5LwC/view?usp=sharing'>
    <img src='https://img.shields.io/badge/Data-GoogleDrive-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Data'>
    </a>
    <a href='https://drive.google.com/file/d/1YDnf0G4eayAzBhTmJSC3kDYfBym57qk8/view?usp=sharing'>
    <img src='https://img.shields.io/badge/Checkpoints-GoogleDrive-green?style=plastic&logo=Google%20chrome&logoColor=green' alt='Checkpoints'>
    </a>
    <a href='https://vimeo.com/1055247476'>
      <img src='https://img.shields.io/badge/Demo-Vimeo-red?style=plastic&logo=Vimeo&logoColor=red' alt='Demo'>
    </a>
</p>

<img src='./images/overview.jpg'>

# Project structure
```
NeuralForceField/
│
├── data/
│   └──iphyre/game_seq_data/
│       ├── activated_pendulum/
│       ├── angle/
│       ├── ...
│       └── support_hole/
│   └── nbody/
│       ├── train_data.npy
│       ├── train_data_long.npy
│       ├── within_data.npy
│       ├── within_data_long.npy
│       └── cross_data.npy
│
├── checkpoints/          
│
├── iphyre/             
│   ├── configs/         # Configuration files
│   ├── models/          # Model dir containing NFF, IN, SlotFormer
│   ├── utils/           # Useful tools such as dataloader
│   ├── planning.py      # Planning script
│   ├── README.md        # An instruction for use
│   ├── test.py          # Evaluation functions
│   └── train.py         # Training functions
│
├── iphyre/             
│   ├── configs/         # Configuration files
│   ├── models/          # Model dir containing NFF, IN, SlotFormer
│   ├── utils/           # Useful tools such as dataloader
│   ├── generate_data.py # Data generation functions
│   ├── planning.py      # Planning script
│   ├── README.md        # An instruction for use
│   ├── test.py          # Evaluation functions
│   └── train.py         # Training functions
```
## Getting started
Make sure you have installed torch, torchdiffeq, iphyre, and rebound.

Go to the specific task directory to train and test the models. The instructions of running commands are provided for each task ([README_iphyre](./iphyre/README.md) and [README_nbody](./nbody/README.md)). Download data [here](https://drive.google.com/file/d/14IQ-DkVIQ0pvI9uXSgrc6j9Wjk1E5LwC/view?usp=sharing) and checkpoints [here](https://drive.google.com/file/d/1YDnf0G4eayAzBhTmJSC3kDYfBym57qk8/view?usp=sharing).
```
cd ./iphyre
```

or

```
cd ./nbody
```
