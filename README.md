# Palm-tree-counting

## Project Overview

The Palm-tree-counting project aims to develop a machine learning application capable of counting palm trees in aerial images with high accuracy. Leveraging pre-trained convolutional neural networks (CNN) like ResNet or EfficientNet, this model undergoes fine-tuning on a dataset of 2,860 images to achieve a validation Root Mean Squared Error (RMSE) of less than 1.5. The project is based on preprocessing data, modifying the CNN, and fine-tuning the model to ensure reliable performance across diverse aerial images.

## Getting Started

### Dependencies

- Python 3
- Flask
- TensorFlow / Keras
- Node.js for running the web interface
- Bootstrap and EJS for frontend

### Installation

1. Install Python dependencies:

```
pip install flask tensorflow
```

2. Navigate to your project directory and install Node.js dependencies:

```
cd path/to/project
npm install
```

### Running the Application

1. Start the backend server:

```
python app.py
```

2. In a new terminal, start the frontend server:

```
npm start
```

Visit `http://localhost:5002` in your browser to access the application.

## Application Structure

- `app.py`: The Flask backend server that handles image prediction requests.
- `package.json` & `index.js`: Setup for the Node.js server and web interface.
- `views/index.ejs`: The frontend web page for uploading images.
- `preprocess_images.py`: Script for preprocessing images for training and validation.
- `modify_cnn_model.py`: Contains logic for modifying and compiling the CNN model.

## Model Training

The model training involves preprocessing the images to a uniform size, normalizing them, and applying data augmentation. The chosen CNN's top layer is replaced with a fully connected layer for regression to output palm tree counts. The training process focuses on minimizing the Mean Squared Error (MSE) to improve the validation RMSE.

## Model Training Instructions

To train the palm tree counting model, ensure you've followed the setup instructions and have the dataset ready in the expected directory structure. Once setup, initiate the training process by running the following command from the project root directory:

```bash
python train.py
```

This script will preprocess your data, set up the model, and begin training, automatically saving the trained model upon completion.

## Contributing

We welcome contributions to the Palm-tree-counting project! Whether it's through submitting issues, writing code, improving documentation, or providing feedback, your input is valued.

## License

This project is licensed under the ISC License. See the LICENSE file for more details.

## Acknowledgements

- Special thanks to the TensorFlow team for their amazing machine learning framework.
- The Flask community for the robust web server framework.