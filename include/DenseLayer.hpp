#ifndef CUSTOMNEURALNETWORKS_DENSELAYER_HPP
#define CUSTOMNEURALNETWORKS_DENSELAYER_HPP

#include <vector>
#include <random>
#include "Layer.hpp"

namespace nn {

class DenseLayer : public Layer {
public:
    DenseLayer(size_t inputSize, size_t outputSize);
    ~DenseLayer() override = default;

    std::vector<float> forward(const std::vector<float>& input) override;
    std::vector<float> backward(const std::vector<float>& gradient) override;
    void updateParameters(float learningRate) override;

    // Save and load methods for model persistence
    void save(nlohmann::json& json) const override;
    void load(const nlohmann::json& json) override;

private:
    size_t inputSize;
    size_t outputSize;
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    std::vector<float> lastInput;
    std::vector<float> lastOutput;
};

} // namespace nn

#endif // CUSTOMNEURALNETWORKS_DENSELAYER_HPP
