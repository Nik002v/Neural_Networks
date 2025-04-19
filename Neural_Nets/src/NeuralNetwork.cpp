#include "NeuralNetwork.h"
#include "Layer.h"
#include "Optimizer.h"
#include "Loss.h"
#include "Regularization.h"

#include <random>
#include <numeric>
#include <iostream>
#include <cmath>

using M = Eigen::MatrixXd;


Sequential::Sequential() {}

void Sequential::add(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

void Sequential::remove_layer(size_t index) {
    if (layers.empty()) {
        throw std::runtime_error("Cannot remove layer from empty model");
    }

    if (index >= layers.size()) {
        throw std::out_of_range("Layer index out of range");
    }

    // Remove the layer
    layers.erase(layers.begin() + index);
}

Eigen::MatrixXd Sequential::forwardpropagation(const M& input) {
    M current = input;
    for (auto& layer : layers) {
        current = layer->forward(current);
    }
    return current;
}

Eigen::MatrixXd Sequential::backpropagation(const M& error)
{
    M current = error;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        auto& layer = *it;
        current = layer->backward(current);
    }
    return current;
}

Eigen::MatrixXd Sequential::predict(const M& input) {
    return forwardpropagation(input);
}

double Sequential::eval(const M& X, const M& y, std::string metrics = "accuracy")
{
    M predictions = forwardpropagation(X);

	if (metrics == "loss") {
		return MSE().compute(predictions,y);
	}

    // Convert predictions to class labels (assuming max probability indicates class)
    M pred_labels = Eigen::MatrixXd::Zero(y.rows(), y.cols());
    for (int i = 0; i < predictions.cols(); i++) {
        Eigen::MatrixXd::Index maxRow;
        predictions.col(i).maxCoeff(&maxRow);
        pred_labels(maxRow, i) = 1;
    }

    // Calculate true positives, false positives, false negatives
    double tp = (pred_labels.array() * y.array()).sum();
    double fp = (pred_labels.array() * (1 - y.array())).sum();
    double fn = ((1 - pred_labels.array()) * y.array()).sum();

    if (metrics == "accuracy") {
        return (pred_labels.array() == y.array()).cast<double>().mean();
    }
    else if (metrics == "precision") {
        return tp / (tp + fp + 1e-10);  // Add small epsilon to avoid division by zero
    }
    else if (metrics == "recall") {
        return tp / (tp + fn + 1e-10);
    }
    else if (metrics == "f1") {
        double precision = tp / (tp + fp + 1e-10);
        double recall = tp / (tp + fn + 1e-10);
        return 2 * (precision * recall) / (precision + recall + 1e-10);
    }
    return 0.0; // Default case
}


void Sequential::train( const M& X,
                        const M& y,
                        Optimizer& optimizer,
                        const Loss& loss_function,
                        const Regularization& regularizer,
                        int epochs,
                        int batch_size)
{
    size_t n_samples = X.cols();
    double epoch_loss_old;
	double T = 1e-3; // Threshold for stopping criterion

    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        int num_batches = 0;

        // Create and shuffle indices
        std::vector<size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));

        // Mini-batch training using indices
        for (size_t i = 0; i < n_samples; i += batch_size) {
            std::pair<M,M> X_batch_y_batch = extract_batch(X, y, indices, i, batch_size);
			M X_batch = X_batch_y_batch.first;
			M y_batch = X_batch_y_batch.second;

            Eigen::MatrixXd predictions = forwardpropagation(X_batch);
            epoch_loss += loss_function.compute(predictions, y_batch);
            M gradient = loss_function.gradient(predictions, y_batch);
            for (const auto& layer : layers) {
                epoch_loss += regularizer.compute(layer->get_weights());
            }
            
            backpropagation(gradient);
            optimizer.update(*this, regularizer);

            num_batches++;
        }
        epoch_loss /= num_batches;
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << ": Loss = " << epoch_loss << std::endl;

        if (epoch != 0 && std::abs(epoch_loss - epoch_loss_old) < T) {
			std::cout << "Training finished because loss update is lower than treshold: " << T << std::endl;
			break;
        }
		epoch_loss_old = epoch_loss;
    }
}


std::pair<M, M> Sequential::extract_batch(
    const M& X,
    const M& y,
    const std::vector<size_t>& indices,
    size_t batch_start,
    size_t batch_size) const
{
    size_t n_samples = X.cols();
    size_t batch_end = std::min(batch_start + batch_size, n_samples);
    size_t current_batch_size = batch_end - batch_start;

    // Create batch matrices
    M X_batch(X.rows(), current_batch_size);
    M y_batch(y.rows(), current_batch_size);

    // Fill batches using indices
    for (size_t j = 0; j < current_batch_size; j++) {
        X_batch.col(j) = X.col(indices[batch_start + j]);
        y_batch.col(j) = y.col(indices[batch_start + j]);
    }

    return { X_batch, y_batch };
}







