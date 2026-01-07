# GennUNet - Abdominal Organ Segmentation

- **Repository:** https://github.com/nicoalpis/GennUNet
- **Dataset:** https://doi.org/10.5281/zenodo.11635577
- **Code Demo**: https://colab.research.google.com/drive/10JyssUcyqbZ9zWPop2fHwdAH5K9LpLe1?usp=sharing
- **Paper:** http://hdl.handle.net/2117/413967

## Model Results

| Organ          | Dice Score (%) |
|:---------------:|:--------------:|
| Spleen       | 97.4         |
| Right Kidney | 96.5         |
| Left Kidney | 96.4         |
| Gallbladder  | 86.8         |
| Esophagus    | 89.0         |
| Liver        | 98.2         |
| Stomach    | 94.2         |
| Aorta    | 96.6         |
| Inferior vena cava    | 93.1         |
| Pancreas     | 89.4         |
| Right adrenal gland    | 84.9         |
| Left adrenal gland    | 85.2         |

## Model Description

GennUNet is a medical image segmentation model for computed tomography (CT) scans. Built on the nnUNet architecture, it achieves high generalizability
across diverse datasets by leveraging a unified dataset from BTCV, AMOS, and TotalSegmentator. The model is optimized to handle variations in imaging properties, 
demographics, and anatomical features, making it robust for real-world clinical applications.

## Model Details

- **Developed by:** Nicolás Álvarez Llopis
- **Supervised by:** María de la Iglesia Vayá, Dario García Gasulla
- **Institution:** Universitat Politècnica de Catalunya (UPC), Universitat de Barcelona (UB), Universitat Rovira i Virgili (URV)
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Architecture:** nnUNet (Fully Convolutional Network)
- **Domain:** Medical Image Segmentation  
- **Modality:** Computed Tomography (CT)  
- **Tasks:** Abdominal Organ Segmentation  
- **Training Framework:** PyTorch, MONAI

## Intended Use

This model is designed for:
- Automated segmentation of abdominal organs in CT scans
- Assisting radiologists in diagnostic workflows
- Medical research involving organ volumetry and disease characterization

## Bias, Risks, and Limitations

The model may be biased in the following ways:

- The model may be biased towards the training data, which primarily consists of publicly available datasets. These datasets do not represent global diversity and may lead to imbalances in model performance across different populations.
- The model may be biased due to sex-based representation imbalances. Historically, medical datasets have overrepresented male subjects, and this study follows the same trend, potentially limiting the model's effectiveness for female patients.
- The model may be biased toward data from specific geographical regions. With most of the data sourced from Europe, North America, and China, populations from South America, Africa, and parts of Asia are underrepresented. This lack of diversity may hinder the model's applicability to a broader range of human anatomical and physiological characteristics.

The model has the following technical limitations:

- The performance of the model may be affected by variations in CT scanners. Differences in imaging quality and characteristics across devices can introduce inconsistencies, limiting the model's generalizability.
- The model's accuracy may degrade over time due to data drift. The training data spans from 2012 to 2021, meaning the anatomical representations used may not fully reflect current patient populations.
- The model's performance may be influenced by contrast enhancement in CT scans. Since the proportion of contrast-enhanced cases in the training dataset is unknown, its impact on prediction quality remains unclear.
- The model is limited by the exclusion of certain anatomical classes. Only classes present across all datasets were included in training, reducing the model's versatility in segmenting a wider range of organs in clinical settings.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

# Load the model
## instantiate the nnUNetPredictor
predictor = nnUNetPredictor(
    tile_step_size=0.5,                 # 50% overlap between adjacent tiles
    use_gaussian=True,                  # Apply Gaussian weighting to smooth tile edges
    use_mirroring=True,                 # Enable test-time augmentation via flipping
    perform_everything_on_device=True,  # Perform all steps (preprocessing, prediction) on GPU
    device=torch.device('cuda', 0),     # Use the first GPU (cuda:0) for computations
    verbose=False,                      # Disable detailed output logs during prediction
    verbose_preprocessing=False,        # Disable logs during preprocessing
    allow_tqdm=True                     # Show progress bar during long tasks
)

## initializes the network architecture, loads the checkpoint
predictor.initialize_from_trained_model_folder(
    "/content/GennUNet/nnUNet_weights",                                # Path to the model weights
    use_folds=(0,1,2,3,4),                                      # Use all 5 folds (for cross-validation)
    checkpoint_name='checkpoint_best.pth',                      # File name of model checkpoints (all must be equal)
)

# Segment CT scan
indir = "/content/GennUNet/input_images"   # Input folder with image files
outdir = "/content/GennUNet/output_images" # Output folder for predictions
predictor.predict_from_files(
    [[join(indir, 'img0027_0000.nii.gz')]],
    [join(outdir, 'img0027_pred.nii.gz')],
    save_probabilities=False,                                   # Do not save the predicted probabilities, just the segmentation
    overwrite=False,                                            # Do not overwrite existing results in the output folder
    num_processes_preprocessing=2,                              # Number of processes for preprocessing
    num_processes_segmentation_export=2,                        # Number of processes for exporting the segmentation
    folder_with_segs_from_prev_stage=None,                      # No previous stage segmentations used
    num_parts=1,                                                # Number of parts to divide the prediction task into
    part_id=0                                                   # ID of the current part (only one part in this case)
)
```

See this [**demo**](https://colab.research.google.com/drive/10JyssUcyqbZ9zWPop2fHwdAH5K9LpLe1?usp=sharing) on how to use the model and visualize its results.

## Training Details

### Training Data

The dataset is available at: https://doi.org/10.5281/zenodo.11635577

GennUNet was trained using a unified dataset consisting of three large-scale abdominal organ segmentation datasets:

| Dataset             | Year | 5-Fold Cross-Val | Test |
|:---------------------:|:------:|:-------:|:---------:|
| BTCV               | 2015 | 30    | 20      |
| AMOS               | 2022 | 272   | 200      |
| TotalSegmentator   | 2023 | 378   | -     |

### Training Procedure

The training code is available at: https://github.com/nicoalpis/GennUNet

#### Preprocessing

**Patch Extraction**

The datasets were processed to remove redundant and inconsistent samples, including intensity normalization, orientation normalization, foreground cropping, and spacing standardization to ensure consistent training input.

**Data Augmentation**

| Technique (MONAI)        | Probability | Range                         |
|:------------------------:|:-----------:|:-----------------------------------------:|
| Rotation               |     0.20    | (-0.52, 0.52)                                         |
| Scaling               |     0.20    | (0.7, 1.4) |
| Gaussian Noise              |     0.10    | (0, 0.1)                    |
| Gaussian Blur             |     0.10    | (0.5, 1.0)                                          |
| Contrast             |     0.15    | (0.75, 1.25)                                           |
| Mirroring          |     0.50 (per axis)    |       |

### Training Hyperparameters

- Loss Function: Dice Loss + Cross-Entropy Loss
- Optimizer: Adam
- Learning Rate: 0.01
- Weight Decay: 0.00003
- Scheduler: PolynomialLR
- Batch Size: 2
- Epochs 1000

  ## Evaluation

The evaluation code is available at: https://github.com/nicoalpis/GennUNet

### Testing Data, Factors & Metrics

#### External Evaluation Data

- [FLARE 2022](https://flare22.grand-challenge.org/)
- [KiTS19](https://kits19.grand-challenge.org/)

#### Metrics

Dice Similarity Coefficient = (2 * TP) / (2 * TP + FP + FN)

### Results

**Validation**

| Dataset           | Dice Score (%) |
|:------------------:|:---------------:|
| BTCV             | 85.97          |
| AMOS             | 90.32         |
| TotalSegmentator | 94.25          |

**Test**

| Dataset           | Dice Score (%) |
|:------------------:|:---------------:|
| BTCV             | 86.17          |
| AMOS             | 90.93         |
| FLARE 2022 | 90.43          |
| KiTS19 | 82.07          |

**Model Performance Comparison**

| Method                | BTCV  | AMOS  | TotalSeg | Arch |
|:-----------------------:|:-------:|:-------:|:----------:|:------:|
| nnUNet (org.)        | 83.08 | 88.64 | 93.20    | CNN  |
| nnUNet ResEnc M      | 83.31 | 88.77 | -        | CNN  |
| nnUNet ResEnc L      | 83.35 | 89.41 | -        | CNN  |
| nnUNet ResEnc XL     | 83.28 | 89.68 | -        | CNN  |
| MedNeXt L k3         | 84.70 | 89.62 | -        | CNN  |
| MedNeXt L k5         | 85.04 | 89.73 | -        | CNN  |
| STU-Net S            | 82.92 | 88.08 | 84.72    | CNN  |
| STU-Net B            | 83.05 | 88.46 | 87.67    | CNN  |
| STU-Net L            | 83.36 | 89.34 | 88.92    | CNN  |
| Swin UNETR           | 78.89 | 83.81 | 84.18    | TF   |
| Swin UNETRV2         | 80.85 | 86.24 | -        | TF   |
| nnFormer             | 80.86 | 81.55 | 79.26    | TF   |
| CoTr                 | 81.95 | 88.02 | -        | TF   |
| No-Mamba Base        | 83.69 | 89.04 | -        | CNN  |
| U-Mamba Bot          | 83.51 | 89.13 | -        | Mam  |
| U-Mamba Enc          | 82.41 | 88.38 | -        | Mam  |
| A3DS SegResNet       | 80.69 | 87.27 | -    | CNN  |
| A3DS DiNTS           | 78.18 | 82.35 | -        | CNN  |
| A3DS SwinUNETR       | 76.54 | 85.05 | -    | TF   |
| Ours (GennUNet)      | **85.97** | **90.32¹** | **94.25²** | CNN  |

¹ Recall that the achieved results with the AMOS dataset lack 3 classes from the original dataset.  
² The exact number of classes to which this study's results are being compared is not specified in the sources. 

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

- **Hardware Type:** V100
- **Hours used:** 1125
- **Hardware Provider:** Joint Research Unit in Biomedical Imaging FISABIO-CIPF
- **Compute Region:** Spain
- **Carbon Emitted:** 62.25kg

## Citation

If you use GennUNet in your research, please cite:
```
@mastersthesis{alvarez2024diverse,
  title={From diverse CT scans to generalization: towards robust abdominal organ segmentation},
  author={{\'A}lvarez Llopis, Nicol{\'a}s},
  year={2024},
  school={Universitat Polit{\`e}cnica de Catalunya}
}
```

---
