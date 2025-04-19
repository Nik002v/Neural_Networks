#include "Layer.h"
#include <random>

Dense::Dense(int input_dim, int output_dim, std::string activation, bool train)
    : Layer(input_dim, output_dim), activation(activation), train(train) {

    weights = Eigen::MatrixXd::Random(output_dim, input_dim + 1) * std::sqrt(2.0 / input_dim + 1);
    w_update = Eigen::MatrixXd::Zero(output_dim, input_dim + 1);
}

Eigen::MatrixXd Dense::forward(const Eigen::MatrixXd& input) {
    Eigen::MatrixXd temp = (weights.leftCols(input_dim) * input).array().colwise() + weights.col(weights.cols() - 1).array();

    last_input = input;
    last_preactivation = temp;
    
    return activation_function(temp, activation);
}

Eigen::MatrixXd Dense::backward(const Eigen::MatrixXd& gradient) {
    Eigen::MatrixXd delta = gradient.array() * activation_derivative(last_preactivation, activation).array();

    Eigen::MatrixXd d_next = weights.leftCols(input_dim).transpose() * delta;
    if (!train) { 
        w_update = Eigen::MatrixXd::Zero(weights.rows(), weights.cols()); 
	    return d_next;
	} 
    Eigen::MatrixXd last_in_aug(last_input.rows() + 1, last_input.cols());
    last_in_aug.topRows(last_input.rows()) = last_input;
    last_in_aug.row(last_in_aug.rows() - 1).setConstant(-1.0);

    w_update = delta * last_in_aug.transpose();

    return d_next;
}

void Dense::update_weights(Eigen::MatrixXd& update) {
    weights -= update;
}

Eigen::MatrixXd Dense::activation_function(const Eigen::MatrixXd& X, std::string activation) const {
    if (activation == "relu") return relu(X);
    if (activation == "sigmoid") return sigmoid(X);
    if (activation == "tanh") return tanh(X);
    if (activation == "softmax") return softmax(X);
    return linear(X);
}

Eigen::MatrixXd Dense::activation_derivative(const Eigen::MatrixXd& X, std::string activation) const {
    if (activation == "relu") return relu_derivative(X);
    if (activation == "sigmoid") return sigmoid_derivative(X);
    if (activation == "tanh") return tanh_derivative(X);
    if (activation == "softmax") return softmax_derivative(X);
    return linear_derivative(X);
}

Eigen::MatrixXd Dense::relu(const Eigen::MatrixXd& X) const {
    return X.array().max(0.0);
}

Eigen::MatrixXd Dense::relu_derivative(const Eigen::MatrixXd& X) const {
    return (X.array() > 0.0).cast<double>();
}

Eigen::MatrixXd Dense::sigmoid(const Eigen::MatrixXd& X) const {
    return 1.0 / (1.0 + (-X.array()).exp());
}

Eigen::MatrixXd Dense::sigmoid_derivative(const Eigen::MatrixXd& X) const {
    Eigen::MatrixXd sig = sigmoid(X);
    return sig.array() * (1.0 - sig.array());
}

Eigen::MatrixXd Dense::tanh(const Eigen::MatrixXd& X) const {
    return X.array().tanh();
}

Eigen::MatrixXd Dense::tanh_derivative(const Eigen::MatrixXd& X) const {
    Eigen::MatrixXd th = tanh(X);
    return 1.0 - th.array().square();
}

Eigen::MatrixXd Dense::softmax(const Eigen::MatrixXd& X) const {
    Eigen::MatrixXd result(X.rows(), X.cols());

    for (int i = 0; i < X.cols(); ++i) {
        Eigen::VectorXd col = X.col(i);

        Eigen::VectorXd stabilized = col.array() - col.maxCoeff();

        Eigen::VectorXd expCol = stabilized.array().exp();
        result.col(i) = expCol / expCol.sum();
    }

    return result;
}

Eigen::MatrixXd Dense::softmax_derivative(const Eigen::MatrixXd& X) const {
    Eigen::MatrixXd s = softmax(X);
    return s.array() * (1.0 - s.array());
}

Eigen::MatrixXd Dense::linear(const Eigen::MatrixXd& X) const {
    return X;
}

Eigen::MatrixXd Dense::linear_derivative(const Eigen::MatrixXd& X) const {
    return Eigen::MatrixXd::Ones(X.rows(), X.cols());
}




