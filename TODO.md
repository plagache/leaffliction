# TODO: Leaffliction

## Project Goals

- Implement an image classification system
- Handle data preprocessing, augmentation, and model training
- Testing the tinygrad framework on an old laptop for efficient neural network training
    - Monitor training time, memory usage, and GPU utilization during the process.
    - Gradually increase the complexity of our models to determine the limitations of our hardware.
    - Understand the fusion kernel techniques that can optimize computation


## Core Tasks

### Data Preparation

- [x] Download data only if necessary / not existing
- [x] Implement `DatasetFolder` class:
    - [x] Find and store class names (directory names).
- [x] Implement `DatasetLoader` class:
    - [x] Add shuffle to dataloader
    - [ ] Implement show_batch functionality
    - [ ] setup train and valid datasets and batches
    - [ ] implement seed to make train and valid dataset the same each time
- [ ] Display one item from each category
- [x] Analyze data distribution
- [ ] gradio analyse tab (1 row for each existing directories)

### Data Augmentation

- [x] Implement data augmentation techniques:
    - [x] Rotate (90 degrees).
    - [x] Flip (randomly).
    - [ ] lower Skew quantity.
    - [x] Shear.
    - [x] Crop (random).
        - [x] Add resize after crop to maintain consistent image sizes
    - [x] Distortion.
- [x] Balance dataset
- [x] Copy Balanced dataset
- [x] Implement sampling for test and training on augmented dataset
- [ ] Review which augmentation to keep for training data
- [ ] gradio tab to pick an image and display augmentation

### Data Transformation

- [ ] Transform dataset to detect features from categories
    - [x] Gaussian Blur.
    - [x] Apply Otsu thresholding to grayscale image.
        - algorithm that automatically determines an optimal threshold value to separate an image into two classes
    - [x] Fill holes in binary image.
    - [x] Apply circular ROI to isolate leaf.
    - [ ] Test pixel intensity to determine threshold
- [x] Implement transformation usage with argparse (handle single file and directory)
    - [ ] when given source and destination directory move image from debug_outdir to destination and rename file accordigly


### Model Training

- [x] Adapt CNN from tinygrad MNIST example for training
    - [x] adapt Training and Validation with new directory/dataset
    - [ ] adapt batch size
    - [ ] adapt Model deepness for our required precision
- [x] Implement function in dataloader to return X_train, Y_train as tinygrad.Tensor
- [ ] Normalize input tensor (/255)
- [ ] Simplify training process (optional debug/context/jit with env variable)
- [ ] Visualize kernel search
- [ ] Train model on prepared data
- [x] Export weights after training
- [ ] train on splitted dataset (train directory)


### Classification

- [ ] Implement classification for new inputs from validation data


### Testing & Validation

- [ ] Test distribution with different directory structures
- [ ] Validate data integrity
- [ ] Check normalization / resize effects

### Predict

- [x] import model/weights
- [x] predict function given an image
- [x] display category/categories and image(optionally transformed)
- [x] display confusion matrix
- [ ] integrate confusion matrix / accuracy with gradio


## Ideas & Questions

### Data Augmentation

Q: What type of augmentation is interesting and natural?
A: Consider augmentations that mimic natural variations:

- Horizontal flips (for most natural objects)
- Slight rotations (within reasonable angles)
- Color jittering (to account for lighting variations)
- Random crops (to focus on different parts of the object)
- Avoid unrealistic transformations (e.g., vertical flips for most natural scenes)


### Architecture Considerations

- Datasets class: Implements item retrieval and counting
- DatasetFolders subclass: Implements item and category retrieval based on folder structure
- DataLoaders class: Takes Datasets class as parameter
- Transform: Function of Datasets
- DataLoaders from fastai does not seem to create/copy new image on disk


### Implementation Details

- Test valid inputs thoroughly
- DataLoaders:
    - get_images: Path to input images
    - get_y: Fetch labels from directory names
    - Implement function to create random validation set
    - Define path to data
    - Specify output data type / models for processing
