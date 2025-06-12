#ifndef CUSTOMNEURALNETWORKS_RELU_HPP
#define CUSTOMNEURALNETWORKS_RELU_HPP

#include "Layer.hpp"
#include <vector>

namespace nn {

class ReLU : public Layer {
public:
    ReLU() = default;
    ~ReLU() override = default;

    std::vector<float> forward(const std::vector<float>& input) override;
    std::vector<float> backward(const std::vector<float>& gradient) override;
    void updateParameters(float learningRate) override {}

    void save(nlohmann::json& json) const override {}
    void load(const nlohmann::json& json) override {}
};

} // namespace nn

#endif // CUSTOMNEURALNETWORKS_RELU_HPP
