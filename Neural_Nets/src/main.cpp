#include "NeuralNetwork.h"
#include "Layer.h"
#include "Optimizer.h"
#include "Loss.h"
#include "Regularization.h"
#include "Preprocessing.h"

#include <iostream>
#include <random>


int main() {

    // 1. Load and preprocess data
    Eigen::MatrixXd data = loadCSV("seq.csv").transpose(); // rowwise examples

    // Create labels vector (first 100 rows class 0, next 100 class 1, etc.)
    const int samples_per_class = 100;
    const int num_classes = 5;
    Eigen::VectorXd labels(data.rows());
    for (int c = 0; c < num_classes; c++) {
        for (int i = 0; i < samples_per_class; i++) {
            labels(c * samples_per_class + i) = c;
        }
    }

    // Split data
    double test_ratio = 0.2;
    Eigen::MatrixXd X_train, X_test;
    Eigen::VectorXd y_train, y_test;
    trainTestSplit(data, labels, test_ratio, X_train, y_train, X_test, y_test);

    // Standardize features
    //standardize(X_train, X_test);

    // One-hot encode labels
    Eigen::MatrixXd y_train_oh = oneHotEncode(y_train, num_classes);
    Eigen::MatrixXd y_test_oh = oneHotEncode(y_test, num_classes);
    
    // Add Gaussian noise to create noisy training data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> noise(0, 1e-1); 

    // Create noisy version of training data
    Eigen::MatrixXd X_train_noisy = X_train;
    for (int i = 0; i < X_train_noisy.rows(); ++i) {
        for (int j = 0; j < X_train_noisy.cols(); ++j) {
            X_train_noisy(i, j) += noise(gen);
        }
    }

    // 2. Create and train autoencoder
    Sequential autoencoder;

    // Add encoder and decoder layers
    autoencoder.add(std::make_unique<Dense>(8000, 32, "relu"));    
	autoencoder.add(std::make_unique<Dense>(32, 8, "relu"));    
    autoencoder.add(std::make_unique<Dense>(8, 32, "relu"));    
    autoencoder.add(std::make_unique<Dense>(32, 8000, "relu"));    

    // Train autoencoder
    auto ae_optimizer = SGD(1e-3);
    auto ae_loss = MSE();
    auto ae_regularizer = L2(1e-4);

    std::cout << "Training autoencoder...\n";
    autoencoder.train(X_train_noisy.transpose(), X_train.transpose(), ae_optimizer, ae_loss, ae_regularizer, 10, 64);

    // Save encoder and decoder weights
    saveMatrixToCSV("encoder_weights1.csv",
        (autoencoder.get_layers()[0])->get_weights());
    saveMatrixToCSV("encoder_weights2.csv",
        (autoencoder.get_layers()[1])->get_weights());

    // 3. Create classification model starting with trained encoder
    Sequential classifier;

    // Add encoder layer (copy from autoencoder)
    classifier.add(std::make_unique<Dense>(8000, 32, "relu",false));
	classifier.add(std::make_unique<Dense>(32, 8, "relu",false));
    // Copy weights from trained encoder
    ((classifier.get_layers()[0])->get_weights()) = loadCSV("encoder_weights1.csv");
	((classifier.get_layers()[1])->get_weights()) = loadCSV("encoder_weights2.csv");

    // Add classification layers
    classifier.add(std::make_unique<Dense>(8, 64, "relu"));
    classifier.add(std::make_unique<Dense>(64, 32, "relu"));
    classifier.add(std::make_unique<Dense>(32, num_classes, "softmax"));

    // Train classifier
    auto clf_optimizer = Adam(1e-1);
    auto clf_loss = CrossEntropy();
    auto clf_regularizer = L2(1e-4);

    std::cout << "Training classifier...\n";
    classifier.train(X_train.transpose(), y_train_oh.transpose(), clf_optimizer, clf_loss,
        clf_regularizer, 20, 32);

    // Evaluate classifier
    double accuracy = classifier.eval(X_test.transpose(), y_test_oh.transpose(), "accuracy");
    std::cout << "Test accuracy: " << accuracy << std::endl; 


    return 0;
}