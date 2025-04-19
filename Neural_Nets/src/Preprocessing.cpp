#include "Preprocessing.h"

#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>


// Train-test split for row-wise examples
void trainTestSplit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y,
                    double testRatio,
                    Eigen::MatrixXd& X_train, Eigen::VectorXd& y_train,
                    Eigen::MatrixXd& X_test, Eigen::VectorXd& y_test)
{

    if (X.rows() != y.rows()) {
        throw std::runtime_error("Number of samples in X and y must match");
    }

    if (testRatio <= 0.0 || testRatio >= 1.0) {
        throw std::runtime_error("Test ratio must be between 0 and 1");
    }

    // Calculate sizes
    int n_samples = X.rows();
    int n_test = static_cast<int>(std::round(n_samples * testRatio));
    int n_train = n_samples - n_test;

    // Create random permutation of indices
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    // Initialize matrices
    X_train = Eigen::MatrixXd(n_train, X.cols());
    X_test = Eigen::MatrixXd(n_test, X.cols());
    y_train = Eigen::MatrixXd(n_train, y.cols());
    y_test = Eigen::MatrixXd(n_test, y.cols());

    // Fill train set
    for (int i = 0; i < n_train; i++) {
        X_train.row(i) = X.row(indices[i]);
        y_train.row(i) = y.row(indices[i]);
    }

    // Fill test set
    for (int i = 0; i < n_test; i++) {
        X_test.row(i) = X.row(indices[n_train + i]);
        y_test.row(i) = y.row(indices[n_train + i]);
    }
}

// Standardize: zero mean, unit variance using training stats (rowwise examples)
void standardize(Eigen::MatrixXd& X_train, Eigen::MatrixXd& X_test) {
    Eigen::RowVectorXd mean = X_train.colwise().mean();
    Eigen::RowVectorXd std = ((X_train.rowwise() - mean).array().square().colwise().mean()).sqrt();
    
    // Prevent division by zero
    for (int i = 0; i < std.size(); ++i) {
        if (std(i) < 1e-6) std(i) = 1;
    }

    X_train = (X_train.rowwise() - mean).array().rowwise() / std.array();
    X_test = (X_test.rowwise() - mean).array().rowwise() / std.array();
}

// One-hot encoding for classification labels
Eigen::MatrixXd oneHotEncode(const Eigen::VectorXd& y, int numClasses) {
    Eigen::MatrixXd oneHot = Eigen::MatrixXd::Zero(y.size(), numClasses);

    for (int i = 0; i < y.size(); ++i) {
        int label = y(i);
        if (label >= 0 && label < numClasses) {
            oneHot(i, label) = 1.0;
        }
        else {
            std::cerr << "Label out of bounds at index " << i << ": " << label << std::endl;
        }
    }
    return oneHot;
}

// Save any Eigen matrix to CSV file
void saveMatrixToCSV(const std::string& filename, const Eigen::MatrixXd& matrix) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            file << matrix(i, j);
            if (j < matrix.cols() - 1)
                file << ",";
        }
        file << "\n";
    }

    file.close();
}


// Load CSV into Eigen::MatrixXd
Eigen::MatrixXd loadCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> values;
    std::string line;

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }

        values.push_back(row);
    }

    if (values.empty()) {
        throw std::runtime_error("CSV file is empty: " + filename);
    }

    // Convert to Eigen::MatrixXd
    size_t rows = values.size();
    size_t cols = values[0].size();
    Eigen::MatrixXd mat(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        if (values[i].size() != cols) {
            throw std::runtime_error("Inconsistent row sizes in CSV file.");
        }

        for (size_t j = 0; j < cols; ++j) {
            mat(i, j) = values[i][j];
        }
    }

    return mat;
}

