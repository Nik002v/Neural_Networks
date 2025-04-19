#include "Optimizer.h"
#include "Regularization.h"
#include "NeuralNetwork.h" 
#include "Layer.h"

void SGD::update(Sequential& nn, const Regularization& regularizer)
{
    for (auto& layer : nn.get_layers()) {
        Eigen::MatrixXd update = learning_rate * layer->get_gradients();
		Eigen::MatrixXd w = layer->get_weights();
		w.col(w.cols() - 1) = Eigen::MatrixXd::Zero(w.rows(), 1); 
        update += learning_rate * regularizer.gradient(w);
        layer->update_weights(update);
    }
}

void Adam::update(Sequential& nn, const Regularization& regularizer) {
    auto& layers = nn.get_layers();

    if (t == 0) {
        for (const auto& layer : layers) {
            Eigen::MatrixXd weights = layer->get_weights();
            m.push_back(Eigen::MatrixXd::Zero(weights.rows(), weights.cols()));
            v.push_back(Eigen::MatrixXd::Zero(weights.rows(), weights.cols()));
        }
    }

    t++;  // Increment timestep

    // Update for each layer
    for (size_t i = 0; i < layers.size(); i++) {
        Eigen::MatrixXd gradient = layers[i]->get_gradients();
        Eigen::MatrixXd w = layers[i]->get_weights();
        w.col(w.cols() - 1) = Eigen::MatrixXd::Zero(w.rows(), 1);
        gradient += regularizer.gradient(w);

        // Update biased first moment estimate
        m[i] = beta1 * m[i] + (1.0 - beta1) * gradient;

        // Update biased second raw moment estimate
        v[i] = beta2 * v[i] + (1.0 - beta2) * gradient.array().square().matrix();

        // Compute bias-corrected first moment estimate
        Eigen::MatrixXd m_hat = m[i] / (1.0 - std::pow(beta1, t));

        // Compute bias-corrected second raw moment estimate
        Eigen::MatrixXd v_hat = v[i] / (1.0 - std::pow(beta2, t));

        // Update parameters
        Eigen::MatrixXd update = learning_rate * m_hat.array() /
            (v_hat.array().sqrt() + epsilon);

        layers[i]->update_weights(update);
    }
}



