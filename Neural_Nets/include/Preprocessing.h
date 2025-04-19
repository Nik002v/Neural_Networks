#pragma once
#include "C:/LIBS/eigen-3.4.0/Eigen/Core"

void trainTestSplit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y,
                    double testRatio,
                    Eigen::MatrixXd& X_train, Eigen::VectorXd& y_train,
                    Eigen::MatrixXd& X_test, Eigen::VectorXd& y_test);

void standardize(Eigen::MatrixXd& X_train, Eigen::MatrixXd& X_test);

Eigen::MatrixXd oneHotEncode(const Eigen::VectorXd& y, int numClasses);

void saveMatrixToCSV(const std::string& filename, const Eigen::MatrixXd& matrix);

Eigen::MatrixXd loadCSV(const std::string& filename);
