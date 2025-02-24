# 🖼️ ML.NET Image Classification Project

## 📌 Project Overview

This project is an **image classification model** built using **ML.NET**. It trains a **deep learning model** to classify images using **ResNetV2-101**. The model is trained on a dataset of images and can predict the category of an image based on its features.

## 🚀 Features

- Uses **ML.NET** for training an image classifier
- Implements **ResNetV2-101** as the pretrained model
- **Loads images** and **trains the model**
- **Validates and predicts** image categories
- Saves the trained model for later inference

## 🛠️ Installation & Setup

### **1️⃣ Clone the Repository**

```sh
git clone https://github.com/your-username/MLNET-ImageClassification.git
cd MLNET-ImageClassification
```

### **2️⃣ Install Python & TensorFlow**

ML.NET requires TensorFlow dependencies. Ensure you have Python installed and TensorFlow properly set up.

1️⃣ **Check Python Version** (must be 3.8 - 3.10):

```sh
python --version
```

2️⃣ **If you don’t have Python 3.10, download & install it:**  
🔗 [Python 3.10 Download](https://www.python.org/downloads/release/python-31012/)

3️⃣ **Install TensorFlow for Python 3.10**

```sh
pip install tensorflow-cpu
```

4️⃣ **Verify TensorFlow Installation**

```sh
python -c "import tensorflow as tf; print(tf.__version__)"
```

Once TensorFlow is installed, proceed with the ML.NET setup below.

### **3️⃣ Install Dependencies**

Ensure you have the required **NuGet packages** installed:

```sh
dotnet add package Microsoft.ML --version 4.0.1
dotnet add package Microsoft.ML.ImageAnalytics --version 4.0.1
dotnet add package Microsoft.ML.TensorFlow --version 4.0.1
dotnet add package Microsoft.ML.Vision --version 4.0.1
dotnet add package SciSharp.TensorFlow.Redist --version 2.3.1
dotnet add package TensorFlow.NET --version 0.20.1.0
```

### **4️⃣ Run the Application**

```sh
dotnet run
```

## 📂 Project Structure

```
MLNET-ImageClassification/
│── Data/                     # Folder containing training images
│── bin/                      # Compiled output (ignored in .gitignore)
│── obj/                      # Temporary build files (ignored in .gitignore)
│── model.zip                 # (Optional) Saved trained model
│── ImageClassification.csproj # Project file
│── Program.cs                 # Main application file
│── README.md                  # Project documentation
│── .gitignore                 # Git ignore rules
```

## 🏗️ How It Works

1. Loads images from the **Data/** folder
2. Preprocesses images using **ML.NET transformations**
3. Trains a **ResNetV2-101** model for classification
4. Evaluates accuracy using a **validation set**
5. Uses the trained model to **predict new images**

## 📊 Example Output

```
AI Predictions:
Image: puppy3.jpg | Actual Label: puppy | Predicted Label: puppy
Image: kitten5.jpg | Actual Label: kitten | Predicted Label: kitten
Image: kitten4.jpg | Actual Label: kitten | Predicted Label: kitten
Image: puppy4.jpg | Actual Label: puppy | Predicted Label: puppy
```

---

🚀 **Happy Coding!** 🎉
