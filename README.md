# Deepfake Image Detection System Using Transfer Learning

## 1. Project Information

**Project Title:** Deepfake Image Detection System Using Transfer Learning

**Team Members:**

- Ayesha Safeer – 01-134242-031
- Laiba Zainab – 01-134242-058

## 2. Project Description

The Deepfake Image Detection System is an AI-based application designed to identify whether an uploaded facial image is real or AI-generated/manipulated. Deepfake images can be difficult to distinguish from authentic images through visual inspection alone.

This project uses a fine-tuned ResNet50 convolutional neural network with transfer learning to analyze images and predict whether they are REAL or FAKE. A user-friendly web application built with Streamlit allows users to upload images, view predictions, and explore visual explanations of the model's results using Grad-CAM.

## 3. Requirements and Installation

### Software Requirements

- Python 3.10 or 3.11 
- pip (Python package installer)
- Internet connection for installing dependencies

### Required Libraries

- Streamlit – Web application interface
- PyTorch – Deep learning framework
- Torchvision – Pretrained models and image transformations
- NumPy – Numerical operations
- Pandas – Data handling and result tables
- Pillow (PIL) – Image processing
- Scikit-image – Image processing utilities
- Matplotlib – Visualization and plotting

### Dataset Information
The project uses the Deepfake and Real Images dataset, available on Kaggle.
Dataset Name: Deepfake and Real Images
Dataset Author: Manjil Karki
Dataset Link: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

### Installation

Open Command Prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

This installs all the required libraries.

## 4. How to Run the Project

The project includes a previously trained model, so users do not need to train it again to run the application.

Follow these steps:

**Step 1: Download or Clone the Project**

Clone the GitHub repository:

```bash
git clone https://github.com/ayeshasafeer1/deepfake-image-detector.git
```

Navigate into the project folder:

```bash
cd deepfake-image-detector
```

**Step 2: Create a Virtual Environment**
```bash
python -m venv .venv
```

**Step 3: Activate the Virtual Environment**
```bash
.venv\Scripts\activate

**Step 4: Install the Required Libraries**

```bash
pip install -r requirements.txt
```

**Step 5: Run the Application**

Execute the following command:

```bash
streamlit run app.py
```

**Step 6: Open the Application**

Streamlit will start a local server and provide a URL, usually:

```text
http://localhost:8501
```

Open this URL in a web browser to access the application.

**Note:** The trained model weights (`best.pt`) and configuration file (`final_settings.json`) must remain inside the `model` folder for the application to work correctly.

## 5. Expected Output

When the application runs successfully, the user will see the Deepfake Image Detector dashboard.

The application provides the following functionality:

- Upload one or multiple images in JPG, JPEG, PNG, or WebP format.
- Analyze uploaded images to classify them as REAL or FAKE.
- View the model's prediction score (P(real)).
- View Grad-CAM visualizations to understand which image regions influenced the prediction.
- Adjust available detection settings through the sidebar.
- View information about the model, its performance, and evaluation results.
- Download prediction results as a CSV file.

The application displays the prediction and related analysis for each uploaded image.

**Live Application:** https://deepfake--image--detector.streamlit.app/

**GitHub Repository:** https://github.com/ayeshasafeer1/deepfake-image-detector

**Training notebook**: https://colab.research.google.com/drive/1igjPtNLr98wQ1XpCcDM4lyJhvZTcdkAT?usp=sharing

## 6. Project Files and Folder Structure

| File/Folder | Purpose |
|---|---|
| `app.py` | Main Streamlit application responsible for the user interface and image analysis workflow. |
| `model_utils.py` | Contains model-related utilities, including loading the trained model and prediction support. |
| `pipeline.py` | Handles image preprocessing, face detection/cropping, and the prediction pipeline. |
| `check_setup.py` | Checks the project environment and setup. |
| `requirements.txt` | Lists all Python libraries required to run the project. |
| `README.md` | Provides project information, installation instructions, and usage guidelines. |
| `.gitignore` | Specifies files and folders that should not be tracked by Git. |
| `.streamlit/` | Contains Streamlit application configuration and theme settings. |
| `assets/` | Contains the application's custom CSS styling. |
| `model/` | Contains the trained model weights (`best.pt`) and model configuration (`final_settings.json`). |
| `samples/` | Reserved for sample images that can optionally be used to test the application. |

## 7. Important Notes

- The trained model is included in the project, so separate model training is not required to run the application.
- The application supports JPG, JPEG, PNG, and WebP image formats.
- The maximum upload size is 20 MB per image.
- Deepfake detection is not completely reliable. Predictions should be treated as an indication rather than absolute proof of authenticity.
- The project is also deployed online using Streamlit Community Cloud.

