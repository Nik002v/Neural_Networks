#pragma once
#include "C:/LIBS/eigen-3.4.0/Eigen/Core"
#include <vector>

class Sequential;
class Regularization;

class Optimizer {
protected:
    double learning_rate;

public:
	Optimizer(double lr = 0.001) : learning_rate(lr) {}
    virtual ~Optimizer() = default;

    virtual void update(Sequential& nn, const Regularization& regularizer) = 0;
};


class SGD : public Optimizer {
public:
    SGD(double lr = 0.01) : Optimizer(lr) {}

    void update(Sequential& nn, const Regularization& regularizer) override;
};


class Adam : public Optimizer {
private:
    double beta1;
    double beta2;
    double epsilon;
    int t;
    std::vector<Eigen::MatrixXd> m;
    std::vector<Eigen::MatrixXd> v;

public:
    Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.99, double eps = 1e-8)
        : Optimizer(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    void update(Sequential& nn, const Regularization& regularizer) override;
};


