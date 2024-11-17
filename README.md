# Melanoma Detection using CNN

## Problem Statement
This project aims to build a CNN-based model for accurately detecting melanoma, a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. The goal is to create a solution that can evaluate images and alert dermatologists about the presence of melanoma, potentially reducing the manual effort needed in diagnosis.

## Table of Contents
1. [Dataset](#dataset)
2. [Project Pipeline](#project-pipeline)
3. [Tools and Libraries Used](#tools-and-libraries-used)
4. [Data Augmentation Techniques](#data-augmentation-techniques)
5. [Model Architecture](#model-architecture)
6. [How to Use](#how-to-use)
7. [Results and Findings](#results-and-findings)
8. [Future Improvements](#future-improvements)
9. [Contributors](#contributors)
10. [License](#license)
11. [Acknowledgments](#acknowledgments)

## Dataset
The dataset consists of 2357 images of malignant and benign oncological diseases, sourced from the International Skin Imaging Collaboration (ISIC). It includes images of nine different skin conditions:

1. Actinic keratosis
2. Basal cell carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented benign keratosis
7. Seborrheic keratosis
8. Squamous cell carcinoma
9. Vascular lesion

The dataset is slightly imbalanced, with melanomas and moles (nevi) being slightly over-represented.

## Project Pipeline

1. **Data Reading/Understanding**: 
   - Define paths for train and test images using os.listdir().
   - Understand the structure of the dataset and print out the number of images in each category.

2. **Dataset Creation**: 
   - Create train & validation datasets using tf.keras.preprocessing.image_dataset_from_directory().
   - Set batch size to 32 and image size to 180x180 pixels.
   - Split the data into 80% training and 20% validation.

3. **Dataset Visualization**: 
   - Use matplotlib to visualize one instance of each class in the dataset.
   - Display a 3x3 grid of images with their corresponding labels.

4. **Model Building & Training**:
   - Create a CNN model using tf.keras.Sequential().
   - Use layers: Conv2D, MaxPooling2D, Dropout, Flatten, and Dense.
   - Apply data augmentation using tf.keras.Sequential() with RandomFlip, RandomRotation, and RandomZoom.
   - Compile the model with Adam optimizer and SparseCategoricalCrossentropy loss.
   - Train for 20 epochs using model.fit().
   - Plot training and validation accuracy/loss curves.

5. **Data Augmentation**: 
   - Implement data augmentation to address overfitting.
   - Use techniques like random flipping, rotation, and zooming.

6. **Model Building & Training on Augmented Data**:
   - Create a new model with the same architecture but including data augmentation layers.
   - Train for 20 epochs and analyze results to see if overfitting is reduced.

7. **Class Distribution Analysis**:
   - Use os.listdir() to count the number of images in each class.
   - Visualize the class distribution using a bar plot.
   - Identify the class with the least samples and the dominant classes.

8. **Handling Class Imbalances**: 
   - Use the Augmentor library to generate additional samples for underrepresented classes.
   - Aim to have at least 500 samples for each class.

9. **Final Model Building & Training**:
   - Create a CNN model using the balanced dataset.
   - Include data augmentation layers in the model.
   - Train for 30 epochs.
   - Plot and analyze final accuracy and loss curves.

## Tools and Libraries Used
- Python 3.x
- TensorFlow 2.x / Keras (with GPU acceleration)
- Augmentor
- Matplotlib and Seaborn (for visualization)
- NumPy (for numerical operations)
- Pandas (for data manipulation)
- PIL (Python Imaging Library)
- Google Colab (for GPU-accelerated training)

## Data Augmentation Techniques
- Random Flipping (horizontal)
- Random Rotation (up to 20 degrees)
- Random Zooming (up to 20%)
- Rescaling: Normalize pixel values to range [0,1]

## Model Architecture
The CNN model consists of:
- Multiple Conv2D layers (starting with 16 filters and increasing)
- MaxPooling2D layers
- Dropout layers for regularization
- Flatten layer
- Dense layers (128 units with ReLU activation)
- Output Dense layer with 9 units (softmax activation)

## How to Use
1. Clone the repository:
   ```
   git clone [repository-url]
   cd [repository-name]
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Open and run the Jupyter notebook `MelanomaDetecting.ipynb`:
   ```
   jupyter notebook MelanomaDetecting.ipynb
   ```
   Or upload and run it in Google Colab for GPU acceleration.
4. Follow the notebook cells to execute each step of the pipeline.

## Results and Findings
[To be filled after model training and evaluation]

Preliminary findings:
- Initial model showed signs of overfitting (high training accuracy, lower validation accuracy).
- Data augmentation helped in reducing overfitting and improving model generalization.
- Class imbalance was identified, with some classes having significantly fewer samples.
- Final model trained on balanced dataset showed improved accuracy across all classes.

## Future Improvements
- Experiment with different CNN architectures (e.g., ResNet, VGG)
- Try transfer learning with pre-trained models on larger datasets
- Collect more real-world data for underrepresented classes
- Implement cross-validation for more robust evaluation
- Explore advanced augmentation techniques specific to medical imaging
- Implement model interpretability techniques (e.g., Grad-CAM) to understand predictions
- Fine-tune hyperparameters for better performance
- Implement ensemble methods to combine predictions from multiple models
- Explore the use of attention mechanisms to focus on relevant image areas
- Investigate the impact of different image preprocessing techniques

## Contributors
Chand Rayee

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- International Skin Imaging Collaboration (ISIC) for providing the dataset
- The developers of TensorFlow, Keras, and Augmentor for their excellent tools
- Google Colab for providing GPU resources for training

## Note
This project is part of an assignment to demonstrate the application of CNNs in medical image classification. The model and findings presented here should not be used for actual medical diagnosis without further validation and approval from medical professionals. The purpose is educational and serves as a starting point for understanding the potential of AI in dermatology.

For any questions, suggestions, or if you'd like to contribute to this project, please open an issue in the repository or contact the contributors. We welcome feedback and collaboration to improve this model and its applications in the field of dermatology.

## Connect With Us 🌐

Feel free to reach out to us through any of the following platforms:

- Telegram: [@chand_rayee](https://t.me/chand_rayee)
- LinkedIn: [Mr. Chandrayee](https://www.linkedin.com/in/mrchandrayee/)
- GitHub: [mrchandrayee](https://github.com/mrchandrayee)
- Kaggle: [mrchandrayee](https://www.kaggle.com/mrchandrayee)
- Instagram: [@chandrayee](https://www.instagram.com/chandrayee/)
- YouTube: [Chand Rayee](https://www.youtube.com/channel/UCcM2HEX1YXcWjk2AK0hgyFg)
- Discord: [AI & ML Chand Rayee](https://discord.gg/SXs6Wf8c)