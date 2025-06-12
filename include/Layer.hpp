#ifndef CUSTOMNEURALNETWORKS_LAYER_HPP
#define CUSTOMNEURALNETWORKS_LAYER_HPP

#include <vector>
#include <nlohmann/json.hpp>

namespace nn {

class Layer {
public:
    virtual ~Layer() = default;

    // Forward pass
    virtual std::vector<float> forward(const std::vector<float>& input) = 0;

    // Backward pass
    virtual std::vector<float> backward(const std::vector<float>& gradient) = 0;

    // Update parameters
    virtual void updateParameters(float learningRate) = 0;

    // Model persistence
    virtual void save(nlohmann::json& json) const = 0;
    virtual void load(const nlohmann::json& json) = 0;
};

} // namespace nn

#endif // CUSTOMNEURALNETWORKS_LAYER_HPP
