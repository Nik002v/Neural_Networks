#pragma once
#include "C:/LIBS/eigen-3.4.0/Eigen/Core"
#include <string>

class Regularization {
public:
    virtual ~Regularization() = default;
    virtual double compute(const Eigen::MatrixXd& weights) const = 0;
    virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd& weights) const = 0;
};

class L1 : public Regularization {
private:
    double lambda;

public:
    explicit L1(double lambda = 0.01) : lambda(lambda) {}

    double compute(const Eigen::MatrixXd& weights) const override {
        return lambda * weights.array().abs().sum();
    }

    Eigen::MatrixXd gradient(const Eigen::MatrixXd& weights) const override {
        return lambda * weights.array().sign().matrix();
    }
};

class L2 : public Regularization {
private:
    double lambda;

public:
    explicit L2(double lambda = 0.01) : lambda(lambda) {}

    double compute(const Eigen::MatrixXd& weights) const override {
        return 0.5 * lambda * weights.array().square().sum();
    }

    Eigen::MatrixXd gradient(const Eigen::MatrixXd& weights) const override {
        return lambda * weights;
    }
};

class ElasticNet : public Regularization {
private:
    double lambda;
    double l1_ratio;

public:
    ElasticNet(double lambda = 0.01, double l1_ratio = 0.5)
        : lambda(lambda), l1_ratio(l1_ratio) {
    }

    double compute(const Eigen::MatrixXd& weights) const override {
        double l1_term = weights.array().abs().sum();
        double l2_term = 0.5 * weights.array().square().sum();
        return lambda * (l1_ratio * l1_term + (1.0 - l1_ratio) * l2_term);
    }

    Eigen::MatrixXd gradient(const Eigen::MatrixXd& weights) const override {
        return lambda * (l1_ratio * weights.array().sign() +
            (1.0 - l1_ratio) * weights.array()).matrix();
    }
};
