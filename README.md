# Brain Tumor MRI Classification

This project uses a deep learning model to classify brain tumors from MRI images. The model is a ResNet-18 convolutional neural network that has been pre-trained on the ImageNet dataset and fine-tuned for this specific task. The goal is to accurately classify MRI images into four categories: glioma, meningioma, pituitary, and no tumor.

## Key Features

- **Transfer Learning**: Utilizes a pre-trained ResNet-18 model to leverage learned features from a large dataset, improving performance and reducing training time.
- **Data Augmentation**: Applies transformations like random resized cropping and horizontal flipping to the training data to increase its diversity and prevent overfitting.
- **Early Stopping**: Monitors the validation accuracy and stops training if there is no improvement for five consecutive epochs, saving time and computational resources.

## Training and Evaluation

The model is trained using the AdamW optimizer and cross-entropy loss with label smoothing. A learning rate scheduler (`ReduceLROnPlateau`) is used to adjust the learning rate during training. The final evaluation is performed on a separate test set, and the results are visualized with a confusion matrix.

## How to Use

1. **Set up the environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Organize the data**: Place the training and testing datasets in the `data/` directory, following the structure below:
   ```
   data/
   ├── Training/
   │   ├── glioma/
   │   ├── meningioma/
   │   ├── notumor/
   │   └── pituitary/
   └── Testing/
       ├── glioma/
       ├── meningioma/
       ├── notumor/
       └── pituitary/
   ```

3. **Run the notebook**: Open and run the `notebook.ipynb` file in the `src/` directory to train the model and see the results.
