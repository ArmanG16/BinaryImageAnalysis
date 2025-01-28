import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC(y, yhat):
    return np.mean(y == yhat)

def measureAccuracyOfPredictors(predictors, X, y):
    predictions = []
    for r1, c1, r2, c2 in predictors:
        predictions.append(X[:, r1, c1] > X[:, r2, c2])
    predictions = np.array(predictions).mean(axis=0) > 0.5
    return fPC(y, predictions)

def stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels):
    m = 6
    predictors = []
    num_samples, height, width = trainingFaces.shape
    for _ in range(m):
        best_accuracy = 0
        best_predictor = None
        for r1 in range(height):
            for c1 in range(width):
                for r2 in range(height):
                    for c2 in range(width):
                        if (r1, c1, r2, c2) in predictors:
                            continue
                        potential_predictors = predictors + [(r1, c1, r2, c2)]
                        accuracy = measureAccuracyOfPredictors(potential_predictors, trainingFaces, trainingLabels)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_predictor = (r1, c1, r2, c2)
        predictors.append(best_predictor)
        print(f"Predictor: {best_predictor} Accuracy: {best_accuracy}")
    test_accuracy = measureAccuracyOfPredictors(predictors, testingFaces, testingLabels)
    print(f"The accuracy with the {m} chosen predictors: {test_accuracy}")
    visualizeFeatures(testingFaces[0], predictors)

#seperated visual features from stepwiseRegression function for self-conveinence reasons
def visualizeFeatures(image, predictors):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    for i, (r1, c1, r2, c2) in enumerate(predictors):
        rect1 = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        rect2 = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
    plt.show()


def loadData(which):
    faces = np.load(f"{which}ingFaces.npy")
    faces = faces.reshape(-1, 24, 24)
    labels = np.load(f"{which}ingLabels.npy")
    return faces, labels


if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels)
