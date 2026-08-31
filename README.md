<h1 align="center">Color Recognition in Challenging Lighting Environments</h1>

<p align="center">
  <a href="https://doi.org/10.1109/I2CT61223.2024.10543537"><img src="https://img.shields.io/badge/Paper-IEEE_I2CT-2f6f9f.svg" alt="Paper"></a>
  <a href="color recognition.py"><img src="https://img.shields.io/badge/Project-color recognition.py-555555.svg" alt="Project entry file"></a>
  <a href="https://scholar.google.com/citations?user=bvyKhaEAAAAJ&hl=en"><img src="https://img.shields.io/badge/Publications-Google_Scholar-4285F4.svg" alt="Google Scholar"></a>
  <a href="https://www.kaggle.com/nizamuddinmaitlo"><img src="https://img.shields.io/badge/Profile-Kaggle-20BEFF.svg" alt="Kaggle profile"></a>
</p>

<p align="center"><b>Nizamuddin Maitlo, Nooruddin Noonari, Sajid Ahmed Ghanghro, Sathishkumar Duraisamy, and Fayaz Ahmed</b></p>

<p align="center">A convolutional neural-network baseline for color classification under lighting variation.</p>

## 🔥 Overview

This repository contains a TensorFlow/Keras CNN baseline for recognizing color categories under varied lighting. The script creates train and test folders, applies image augmentation, trains a convolutional model, and reports classification metrics and visual diagnostics.

## ✨ Features

- Folder-based dataset splitting.
- Image augmentation for improved robustness.
- Convolutional architecture trained with categorical cross-entropy.
- Accuracy and loss curves, confusion matrix, and classification report.

## 🧪 Method and protocol

- The source dataset must be organized with one folder per color class.
- The script creates an 80/20 train-test split before fitting the model.
- Training augmentation is applied only through the training generator.
- Edit the three path variables near the top of `color recognition.py` before running.

## 📁 Repository contents

| File | Purpose |
|---|---|
| `color recognition.py` | Dataset splitting, CNN training, evaluation, and plots |

## 🛠️ Setup

Install the dependencies:

~~~bash
python -m pip install tensorflow numpy matplotlib seaborn scikit-learn
~~~

## 📦 Data and inputs

| Resource | Purpose | Availability |
|---|---|---|
| Different Colors in Challenging Lightening | Original color-classification image collection | [Kaggle dataset](https://www.kaggle.com/datasets/nizamuddinmaitlo/different-colors-in-challenging-lightening) |
| Different Colors in Challenging Lightening v2 | Expanded color × illumination collection for newer experiments | [Kaggle dataset](https://www.kaggle.com/datasets/nizamuddinmaitlo/different-colors-in-challenging-lightening-v2) |

The current script expects a single class-folder hierarchy. If using the v2 color/illumination hierarchy, adapt the indexing logic or select the intended folder level.

## 🚀 Running the project

Set `dataset_dir`, `train_dir`, and `test_dir` in the script, then run:

~~~bash
python "color recognition.py"
~~~

## ♻️ Reproducibility

- Record the Python and library versions used for each run.
- Keep preprocessing, splits, thresholds, and random seeds fixed when comparing results.
- Do not commit private input data, generated model weights, or machine-specific paths.
- Revalidate results when the dataset, sensor, operating environment, or dependency versions change.

## 📚 Paper information

This repository contains the CNN research line associated with the published challenging-light color-recognition study.

| Publication | Venue | Link |
|---|---|---|
| Color Recognition in Challenging Lighting Environments: CNN Approach | 2024 IEEE 9th International Conference for Convergence in Technology (I2CT), 1–7 | [DOI](https://doi.org/10.1109/I2CT61223.2024.10543537) |

## ⭐ Citation

~~~bibtex
@inproceedings{maitlo2024color,
  title     = {Color Recognition in Challenging Lighting Environments: CNN Approach},
  author    = {Maitlo, Nizamuddin and Noonari, Nooruddin and Ghanghro, Sajid Ahmed and Duraisamy, Sathishkumar and Ahmed, Fayaz},
  booktitle = {2024 IEEE 9th International Conference for Convergence in Technology (I2CT)},
  pages     = {1--7},
  year      = {2024},
  doi       = {10.1109/I2CT61223.2024.10543537}
}
~~~

A machine-readable [CITATION.cff](CITATION.cff) file is included for GitHub's citation interface.



## ⚠️ Scope and limitations

The script uses local absolute paths and a simple random file split. Before reporting new results, use dataset-relative paths, verify that near-duplicate images do not cross splits, and evaluate held-out illumination conditions separately.

## 📄 License

No standalone code-license file is currently included in this repository. Dataset and publication terms remain separate.

## 🤝 Acknowledgements

This project uses open-source Python libraries and the data or inputs described above. We thank the original dataset, framework, and software contributors.
