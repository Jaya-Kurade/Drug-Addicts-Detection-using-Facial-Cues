# Drug-Addicts-Detection-using-Facial-Cues
Addiction rates are increasing worldwide, bringing serious health and social consequences. Recent studies have shown that certain facial characteristics can serve as indicators, or biomarkers, of addiction. This opens up an opportunity for AI and ML models to play a key role- offering a solution that's lightweight, portable, and easy to deploy for early detection.

This project explores AI-driven early detection of drug addiction using facial image analysis. Traditional methods are slow and intrusive, prompting the use of deep learning for rapid, non-invasive screening. Using a balanced Kaggle dataset (560 images), we built two models: a custom CNN and a VGG16-based transfer learning classifier. Preprocessing included resizing and real-time augmentation. Trained in Keras and deployed via Gradio, the best model achieved **68%** **test accuracy** showing moderate potential. Future work will focus on improving data quality, model architecture, and regularization techniques. 

**Environment Setup**

Import Deep learning and utility libraries (TensorFlow/Keras, NumPy, Matplotlib, Seaborn, scikit-learn, Gradio).

Mount Google Drive in Colab:

from google.colab import drive
drive.mount('/content/drive')

**Data Loading and Directory Structure**
Organize dataset in this format: Define train, validation, and test directory paths pointing to the image dataset folders inside Google Drive.
addict_dataset/
    drug_addict/
        addicted/
        nonaddicted/
Specify data directories in script: Use Keras ImageDataGenerator.flow_from_directory to load images from these folders, resize them to 128×128, and assign binary labels based on subfolder names
train_dir = '/content/drive/MyDrive/addict_dataset/drug_addict'
val_dir = '/content/drive/MyDrive/addict_dataset/drug_addict'
test_dir = '/content/drive/MyDrive/addict_dataset/drug_addict'

**Data Preprocessing and Augmentation**
Define generators: Normalize pixel values by rescaling all images to the range for more stable training.
Apply data augmentation on the training set (rotations, shifts, shear, zoom, horizontal and vertical flips) to synthetically increase dataset diversity and reduce overfitting

**Baseline CNN Model (From Scratch)**
Build the model: Build a Sequential CNN with multiple Conv2D → BatchNormalization → MaxPooling2D blocks to learn hierarchical visual features from the input images.
Flatten the extracted feature maps, pass them through dense layers with Dropout for regularization, and output a single sigmoid neuron for binary classification (Addicted vs Non-Addicted).

**Model Compilation and Initial Training**
Compile and train model: Compile the baseline CNN using the Adam optimizer, binary cross-entropy loss, and accuracy as the main evaluation metric.
Train the model on the augmented training data for a fixed number of epochs while validating on the validation set to monitor generalization performance

**Evaluation and Training Curves**
Evaluate accuracy and plot curves: Evaluate the trained CNN on the test set to obtain final test accuracy and loss.
Plot training and validation accuracy/loss curves to visualize convergence behavior and detect underfitting or overfitting patterns across epochs.

**Confusion Matrix and Classification Report**
Make predictions and print metrics: Generate predictions on the test set and convert predicted probabilities into binary class labels using a threshold (e.g., 0.5).
Compute and display the confusion matrix and classification report (precision, recall, F1-score) to understand class-wise performance and typical misclassifications

**Training improvements and callbacks**
Add EarlyStopping to monitor validation loss, automatically stopping training when there is no improvement for several epochs and restoring the best weights.​
Use ModelCheckpoint to save the best-performing model (based on validation accuracy), and optionally reduce the learning rate to stabilize training.

**Transfer Learning with VGG16**
Load base model, add custom head, compile: Load a pre-trained VGG16 model (without the top classification layers) initialized with ImageNet weights and freeze its convolutional layers to reuse generic visual features.
Add custom head layers (GlobalAveragePooling2D, Dense, Dropout, and a final sigmoid output) on top of VGG16 to adapt these features to the addiction classification task, then compile the new model.

**Sample Prediction Visualization**
Visualize predictions: Take a batch of images from the test generator and run predictions using the VGG16-based model.
Plot a 3×3 grid of sample images, annotating each with its true label and predicted label to qualitatively inspect model behavior and typical successes/failures. 

**Model Saving and Reloading**
Save/load model: Save the trained model to disk as addiction_detector_model.h5 so it can be reused later without retraining.
Reload the saved model in a new session or script to perform inference or to integrate it into other applications

**Gradio Web Demo**
Install Gradio 
!pip install gradio
Create and launch a Gradio Interface with an image input and label output so users can upload images through a simple web UI and see real-time model predictions.

**How to Run?**
1. Clone this repository and upload your dataset to Google Drive.
2. Open the notebook in Google Colab.
3. Run all cells section-by-section following the above workflow.
4. Deploy the Gradio app to test the trained model interactively.
