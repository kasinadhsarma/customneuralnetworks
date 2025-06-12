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
    input_cache = input;
    std::vector<float> output(input.size());

    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        // Apply leaky ReLU with small slope for negative values
        output[i] = input[i] > 0.0f ? input[i] : 0.01f * input[i];
    }

    return output;
}

std::vector<float> ReLU::backward(const std::vector<float>& gradient) {
    std::vector<float> output(gradient.size());

    #pragma omp parallel for
    for (size_t i = 0; i < gradient.size(); ++i) {
        // Leaky ReLU derivative
        output[i] = gradient[i] * (input_cache[i] > 0.0f ? 1.0f : 0.01f);
    }

    return output;
}

void ReLU::update(float learning_rate) {
    // ReLU has no parameters to update
}

} // namespace nn
