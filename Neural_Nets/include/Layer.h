#pragma once

#include "C:/LIBS/eigen-3.4.0/Eigen/Core"
#include <string>

class Layer {
protected:
    int input_dim;
    int output_dim;

public:
    Layer(int input_dim, int output_dim) : input_dim(input_dim), output_dim(output_dim) {}
    virtual ~Layer() = default;

    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient) = 0;

    virtual Eigen::MatrixXd& get_gradients() = 0;
    virtual Eigen::MatrixXd& get_weights() = 0;

    virtual void update_weights(Eigen::MatrixXd& update) = 0;
};


class Dense : public Layer {
private:
    Eigen::MatrixXd weights;
    Eigen::MatrixXd w_update;

    std::string activation;
	bool train = true;

    Eigen::MatrixXd last_input;
    Eigen::MatrixXd last_preactivation;

public:
    Dense(int input_dim, int output_dim, std::string activation = "relu", bool train = true);

    // Override base class virtual functions
    Eigen::MatrixXd& get_weights() override { return weights; }
    const Eigen::MatrixXd& get_weights() const { return weights; }

    Eigen::MatrixXd& get_gradients() override { return w_update; }

    // Forward and backward passes for batch processing
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient) override;

    void update_weights(Eigen::MatrixXd& update) override;

private:
    // Activation functions for batch processing
    Eigen::MatrixXd relu(const Eigen::MatrixXd& X) const;
    Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd& X) const;

    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& X) const;
    Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd& X) const;

    Eigen::MatrixXd tanh(const Eigen::MatrixXd& X) const;
    Eigen::MatrixXd tanh_derivative(const Eigen::MatrixXd& X) const;

    Eigen::MatrixXd softmax(const Eigen::MatrixXd& X) const;
    Eigen::MatrixXd softmax_derivative(const Eigen::MatrixXd& X) const;

    Eigen::MatrixXd linear(const Eigen::MatrixXd& X) const;
    Eigen::MatrixXd linear_derivative(const Eigen::MatrixXd& X) const;

    // Activation function selectors
    Eigen::MatrixXd activation_function(const Eigen::MatrixXd& X, std::string activation) const;
    Eigen::MatrixXd activation_derivative(const Eigen::MatrixXd& X, std::string activation) const;
};
