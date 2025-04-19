#pragma once
#include "C:/LIBS/eigen-3.4.0/Eigen/Core"

class Loss {
public:
    virtual ~Loss() = default;
    virtual double compute(const Eigen::MatrixXd& y_pred,
        const Eigen::MatrixXd& y_true) const = 0;
    virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd& y_pred,
        const Eigen::MatrixXd& y_true) const = 0;
};

class MSE : public Loss {
public:
    double compute(const Eigen::MatrixXd& y_pred,
        const Eigen::MatrixXd& y_true) const override {

        return (y_pred - y_true).squaredNorm() / (2.0 * y_pred.cols());
    }

    Eigen::MatrixXd gradient(const Eigen::MatrixXd& y_pred,
        const Eigen::MatrixXd& y_true) const override {
        return (y_pred - y_true) / y_pred.cols();
    }
};

class CrossEntropy : public Loss {
public:
    double compute(const Eigen::MatrixXd& y_pred,
        const Eigen::MatrixXd& y_true) const override {
        // Average loss over batch
        return -(y_true.array() * y_pred.array().log()).sum() / y_pred.cols();
    }

    Eigen::MatrixXd gradient(const Eigen::MatrixXd& y_pred,
        const Eigen::MatrixXd& y_true) const override {
        return -y_true.array() / y_pred.array();
    }
};

