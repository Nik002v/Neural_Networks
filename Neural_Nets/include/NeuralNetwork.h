#pragma once

#include "C:/LIBS/eigen-3.4.0/Eigen/Core"
#include <vector>

class Layer;
class Loss;
class Optimizer;
class Regularization;


class NeuralNetwork {
public:
    virtual ~NeuralNetwork() = default;

    virtual void train(const Eigen::MatrixXd& X,
                        const Eigen::MatrixXd& y,
                        Optimizer& optimizer,
                        const Loss& loss_function,
                        const Regularization& regularizer,
                        int epochs,
                        int batch_size) = 0;

    virtual Eigen::MatrixXd predict(const Eigen::MatrixXd& input) = 0;

    virtual double eval(const Eigen::MatrixXd& X,
                        const Eigen::MatrixXd& y,
                        std::string metrics) = 0;
};


class Sequential : public NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    Sequential();

    std::vector<std::unique_ptr<Layer>>& get_layers() { return layers; }
    const std::vector<std::unique_ptr<Layer>>& get_layers() const { return layers; }

    void add(std::unique_ptr<Layer> layer);
    void remove_layer(size_t index);

    Eigen::MatrixXd predict(const Eigen::MatrixXd& input) override;

    double eval(const Eigen::MatrixXd& X,
                const Eigen::MatrixXd& y,
                std::string metrics) override;

    void train(const Eigen::MatrixXd& X,
               const Eigen::MatrixXd& y,
               Optimizer& optimizer,
               const Loss& loss_function,
               const Regularization& regularizer,
               int epochs,
               int batch_size) override;

private:
    Eigen::MatrixXd forwardpropagation(const Eigen::MatrixXd& input);
    Eigen::MatrixXd backpropagation(const Eigen::MatrixXd& gradient);

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> extract_batch(
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& y,
        const std::vector<size_t>& indices,
        size_t batch_start,
        size_t batch_size) const;
};







