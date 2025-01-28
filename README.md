# Step-wise Ensemble Classification for Binary Image Analysis

This project focuses on implementing a step-wise ensemble classifier that analyzes grayscale images to classify whether the image depicts a smile (1) or not (0). The classification is performed using simple binary comparisons between pixel values and selecting features using a greedy algorithm.

## 1. Step-wise Classification

### Description:
This Python program trains a smile classifier using step-wise feature selection. It analyzes grayscale images of size pixels and outputs a prediction indicating whether the image is smiling or not. The classifier selects features (pixel comparisons) based on maximizing accuracy during each step.

### Features:
- Selects up to 6 optimal features using a greedy algorithm.
- Utilizes binary comparisons between pixel intensities as the features.
- Measures training and testing accuracy for subsets of the training data.
- Visualizes the selected features on a sample test image.

### Tasks Implemented:
1. **Feature Selection**:
   - Selects features by iteratively evaluating all possible pixel comparisons to find those that maximize training accuracy.
   - Ensures vectorized computations for efficiency.

2. **Accuracy Evaluation**:
   - Evaluates both training and testing accuracy for various training set sizes (e.g., 400, 600, ..., 2000 samples).
   - Outputs a table summarizing the relationship between training size, training accuracy, and testing accuracy.

3. **Visualization**:
   - Displays the learned features by highlighting pixel locations involved in comparisons on a test image.

## What I Learned:

### Step-wise Feature Selection:
- **Feature Selection Process**: Gained insights into how a greedy algorithm can iteratively select the most effective features to maximize classifier accuracy.
- **Binary Comparisons**: Learned how to use simple comparisons between pixel values as effective features for classification.

### Machine Learning Fundamentals:
- **Ensemble Methods**: Explored ensemble classification techniques by averaging predictions across multiple predictors.
- **Training and Testing**: Understood how to evaluate the impact of training size on model generalization and performance.

### Python Programming:
- **Vectorized Computations**: Improved skills in leveraging NumPy for efficient computations over large datasets.
- **Data Visualization**: Used Matplotlib to highlight and visualize key features learned by the classifier.

This project demonstrates the practical implementation of a step-wise feature selection method for binary image classification, reinforcing both machine learning and programming concepts.
