# Brain Tumour Identification

This project uses a deep learning model to accurately identify brain tumors from MRI scans.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models](#models)
- [Usage](#usage)
- [Results](#results)

## Project Overview

The goal of this project is to build a model that can classify different types of brain tumors from MRI images. The primary notebook for this project is `main.ipynb`, which contains the complete workflow from data preprocessing to model training and evaluation. The `helper.ipynb` notebook contains utility functions used in the main notebook.

## Dataset

The dataset is located in the `Data/` directory and is split into `train` and `test` sets. The images are categorized into different types of tumors, such as glioma, meningioma, etc.

## Models

The trained models are saved in the `Models/` directory. There are two versions of the model:

- `final_model_with_text.keras`: A model that incorporates textual information along with the images.
- `final_model_without_text.keras`: A model that only uses the images for classification.

Visualizations of the model architectures are also available in the `Models/` directory.

## Usage

To run this project, you will need to have Python and the necessary libraries installed. The main script is the `main.ipynb` notebook, which can be run in a Jupyter environment.

## Results

The performance of the models is documented in the `Logs/` and `Metrics/` directories. These include training logs, confusion matrices, and plots of loss and accuracy.
