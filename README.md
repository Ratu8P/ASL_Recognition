# ASL_Recognition

## Project Overview
This project focuses on developing a hand gesture recognition system for American Sign Language (ASL) using keypoints extracted with MediaPipe and processed by a lightweight Convolutional Neural Network (CNN). By leveraging 3D keypoint data instead of raw images, the system significantly reduces input dimensionality while preserving critical spatial information for classification. The proposed method ensures real-time performance, making it suitable for applications in assistive technology and human-computer interaction.

## Dataset 
Link: https://www.kaggle.com/datasets/grassknoted/asl-alphabet


### Features
- **Preprocessing:** The `preprocess.py` script handles data preparation and cleaning.
- **Model Training:** The `train.py` script facilitates model training, utilizing the architecture defined in `model.py`.
- **Deployment:** The `main.py` script provides functionalities to test the trained model, using the weights from `keypoint_model_L_cnn_50.pth`.
- **Data Loading:** Utilities in `data_loader.py` streamline data handling.


### Usage
#### Dependencies
Install the required dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
#### Preprocessing
Run the preprocessing script to clean and format the data:
```bash
python preprocess.py
```

#### Training
Initiate model training using the following command:
```bash
python train.py
```

#### Testing and Inference
Evaluate or use the model for predictions:
```bash
python main.py
```

### Files Overview
- **`class_mapping.txt`**: Provides class mappings for the dataset.
- **`train.py`**: Script for training the model.
- **`preprocess.py`**: Data preprocessing utility.
- **`main.py`**: Main entry point for testing and deployment.
- **`model.py`**: Defines the model architecture.
- **`data_loader.py`**: Handles data loading operations.
- **`keypoint_model_L_cnn_50.pth`**: Pre-trained model weights.

1. Data preprocess and split to train, valid, test
   * "preprocess.py"
   1. change line 89-91 in "preprocess.py"
       * dataset_dir = the path contains folder "A", "B", "C"
       * output_dir = customizable
       * split_dir = customizable, will be used for training
    2. run this file
2. Train
   * "train.py"
   1. change line 14-16 to your split_dir path with /train , /valid, /test
   2. run this file
3. Run
   * "main.py"
   1. change model path on line 13
   2. run 
   this model only works on left hands since all data sample are photos of left hands



