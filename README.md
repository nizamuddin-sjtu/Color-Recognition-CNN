# Color-Recognition-CNN
Color-Recognition-CNN code and link to dataset
Description:

This repository contains a comprehensive implementation of a Convolutional Neural Network (CNN) designed to recognize different colors in challenging lighting conditions. The model is trained on a custom dataset that includes images of various colors captured under diverse lighting environments. The primary objective is to accurately classify the colors, even when the lighting conditions vary significantly.

Key Features:

    Data Preprocessing: Includes scripts for splitting the dataset into training and testing sets, along with image augmentation techniques to enhance the robustness of the model.

    CNN Model Architecture: A carefully designed CNN with multiple convolutional layers, max-pooling, and dense layers. The model is optimized using the Adam optimizer and is trained to minimize categorical cross-entropy loss.

    Model Evaluation: The model's performance is evaluated using accuracy metrics, a confusion matrix, and a classification report. The repository also includes code for plotting the confusion matrix as a heatmap for better visualization.

    Visualization: Implements graphical representations of the model's performance, including accuracy and loss curves, as well as a confusion matrix heatmap.

    Dependencies: The project is built using Python, TensorFlow, Keras, NumPy, Matplotlib, Seaborn, and Scikit-learn.

Dataset: The dataset used in this project is available on Kaggle and can be found here. The dataset contains images of various colors under different lighting conditions.

Usage:

    Clone the repository and navigate to the project directory.
    Install the required dependencies using pip install -r requirements.txt.
    Follow the instructions in the Jupyter notebook or Python scripts to train the model on the provided dataset.
    Evaluate the model's performance using the included test scripts and visualize the results.

Citation:

If you use this code or dataset in your research, please cite the following paper:

N. Maitlo, N. Noonari, S. A. Ghanghro, S. Duraisamy, and F. Ahmed, "Color Recognition in Challenging Lighting Environments: CNN Approach," 2024 IEEE 9th International Conference for Convergence in Technology (I2CT), Pune, India, 2024, pp. 1-7, doi: 10.1109/I2CT61223.2024.10543537.

Keywords: Image segmentation, Computer vision, Image color analysis, Image edge detection, Neural networks, Lighting, Object segmentation, Deep Learning, Convolutional Neural Network (CNN), Image Segmentation, Color Detection, Object Segmentation

Applications:

This model can be used in scenarios where accurate color recognition is essential despite challenging lighting conditions, such as in computer vision systems for robotics, autonomous vehicles, and augmented reality (AR) applications.

License:

This project is licensed under the MIT License - see the LICENSE file for details.
