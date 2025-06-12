#include "NeuralNetwork.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace nn {

float ReLU::activate(float x) {
    return std::max(0.0f, x);
}

float ReLU::derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

std::vector<float> ReLU::forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(), [](float x) {
        return std::max(0.0f, x);
    });
    return output;
}

std::vector<float> ReLU::backward(const std::vector<float>& gradient) {
    std::vector<float> output(gradient.size());
    std::transform(gradient.begin(), gradient.end(), output.begin(), [](float x) {
        return x > 0 ? 1.0f : 0.0f;
    });
    return output;
}

void ReLU::update(float learning_rate) {
    // ReLU has no parameters to update
}

} // namespace nn
