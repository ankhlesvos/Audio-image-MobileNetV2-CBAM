# Audio-image-MobileNetV2-CBAM: A Lightweight and Flexible CNN Framework for Sound and Image Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is the implementation for my graduation thesis: **"Research on Lightweight CNN for Underwater Acoustic Target Classification Optimized by Attention Mechanism"**. It features a highly flexible, custom-built CNN model, `MyNet`, based on the MobileNetV2 architecture, and explores the effectiveness of the CBAM attention mechanism for classification tasks.

While initially designed for acoustic classification on the `ShipsEar` dataset, this framework has been refactored to be a general-purpose tool, also demonstrating strong performance on standard image classification benchmarks like `CIFAR-10`.

**This is a work-in-progress project.** The core modules and experiment pipeline are complete, and further optimizations and analyses are ongoing.

## Core Features & Innovations

*   **Flexible `MyNet` Architecture**: A custom MobileNetV2-like model where attention mechanisms (`CBAM`) can be dynamically inserted at different stages and positions (`pre-dw` vs. `post-dw`) simply through a YAML config file.
*   **Systematic Ablation Studies**: The project is structured to easily conduct ablation studies. An experiment management script (`run_experiments.py`) automates the process of generating configs and running multiple training sessions.
*   **Domain Agnostic**: Includes separate, clean data pipelines and training scripts for both **audio (`train.py`)** and **image (`train_image.py`)** classification tasks, demonstrating the model's versatility.
*   **Key Finding (Preliminary)**: Initial experiments on CIFAR-10 suggest that placing the CBAM attention module **before** the depthwise convolution (`pre-dw`) in the middle stages of the network yields the best performance, effectively balancing feature enhancement and preventing attention overfitting.

### Preliminary Results of Ablation Studies on CIFAR-10 (50 Epochs)

The following table summarizes the performance of different CBAM attention deployment strategies within our `MyNet` architecture. The `pre_dw` strategy, which applies attention *before* the depthwise convolution, generally outperforms the `post_dw` strategy.

The `preDW_s24` configuration, applying attention only at the critical downsampling stages (Stage 2 and 4), emerged as the champion model.

| Model Configuration        | Attention Strategy | Accuracy | Notes                               |
| :------------------------- | :----------------- | :------- | :---------------------------------- |
| **`mynet_baseline`**       | No Attention       | **0.8467** |  |
|                            |                    |          |                                     |
| **Post-DW Attention Series** |                    |          |             |
| `cifar10s_postDW_s4`       | Post-DW @ Stage 4  | **0.8686** | Best performer in the Post-DW group.    |
| `cifar10s_postDW_s24`      | Post-DW @ Stage 2, 4| 0.8665   |                                     |
| `cifar10s_postDW_s45`      | Post-DW @ Stage 4, 5| 0.8633   | Performance degrades with more attention. |
| `cifar10s_postDW_s345`     | Post-DW @ Stage 3, 4, 5| 0.8652   |                                     |
| `cifar10s_postDW_s2345`    | Post-DW @ Stage 2-5| 0.8649   |                                     |
|                            |                    |          |                                     |
| **Pre-DW Attention Series**  |                    |          |           |
| **`cifar10s_preDW_s24`**   | Pre-DW @ Stage 2, 4| **0.8725** | **Overall Best Performance.**       |
| `cifar10s_preDW_s2345`     | Pre-DW @ Stage 2-5 | 0.8696   | Second best, but more complex.      |
| `cifar10s_preDW_s4`        | Pre-DW @ Stage 4   | 0.8676   |                                     |
| `cifar10s_preDW_s345`      | Pre-DW @ Stage 3, 4, 5| 0.8662   |                                     |
| `cifar10s_preDW_s45`       | Pre-DW @ Stage 4, 5| 0.8647   |                                     |

*Note: `sX` denotes that attention is applied in Stage X of the network.*

## Getting Started

### 1. Prerequisites

*   Python 3.9+
*   PyTorch (GPU version recommended)
*   Anaconda

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/ankhlesvos/Audio-image-MobileNetV2-CBAM.git
cd TinyAudioNet

# Create and activate conda environment
conda create -n TinyAudioNet python=3.9
conda activate TinyAudioNet

# For pip（It's better to install pytorch by the guide on pytorch.org)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or for conda
conda install pytorch torchvision torchaudio pytorch-cuda=xx.x -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Preparation

*   **For CIFAR-10**: Simply run the conversion script. It will automatically download and format the dataset.
    ```bash
    python convert_cifar10.py
    ```
*   **For ShipsEar**: Place your dataset (organized by class folders) in the `data/ShipsEar/` directory, then run the list generation script.
    ```bash
    python create_data_lists.py
    ```

### 4. Usage

#### Training

To run the ablation studies for CIFAR-10, use the experiment management script:
```bash
python run_experiments.py
```

To run a single training session for the audio task:
```bash
python train.py -c configs/your_audio_config.yml
```

#### Evaluation & Testing

Use the `test.py` script to get a comprehensive performance report, including Precision, Recall, F1-score, and performance curves.
```bash
python test.py -c path/to/config.yml -m path/to/best_model.pth
```

## Acknowledgements

This project's initial structure and some utility functions were inspired by the `AudioClassification-Pytorch` repository. I extend my gratitude to its author for providing a valuable starting point.

The core model, `MyNet`, is a custom implementation based on the principles introduced in the following academic paper:

*   **MobileNetV2**: Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

The datasets used in this research are publicly available:

*   **CIFAR-10**: Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
