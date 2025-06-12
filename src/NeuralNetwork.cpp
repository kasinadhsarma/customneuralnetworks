#include "NeuralNetwork.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace nn {

DenseLayer::DenseLayer(size_t input_size, size_t output_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / input_size));

    weights.resize(input_size, std::vector<float>(output_size));
    biases.resize(output_size, 0.0f);
    
    for (auto& row : weights) {
        for (float& weight : row) {
            weight = dist(gen);
        }
    }
}

std::vector<float> DenseLayer::forward(const std::vector<float>& input) {
    input_cache = input;
    std::vector<float> output(weights[0].size(), 0.0f);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < weights[0].size(); ++i) {
        for (size_t j = 0; j < input.size(); ++j) {
            output[i] += input[j] * weights[j][i];
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] += biases[i];
    }

    return output;
}

std::vector<float> DenseLayer::backward(const std::vector<float>& gradient) {
    weight_gradients.resize(weights.size(), std::vector<float>(weights[0].size(), 0.0f));
    bias_gradients = gradient;

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[0].size(); ++j) {
            weight_gradients[i][j] = input_cache[i] * gradient[j];
        }
    }

    std::vector<float> input_gradient(input_cache.size(), 0.0f);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < input_cache.size(); ++i) {
        for (size_t j = 0; j < gradient.size(); ++j) {
            input_gradient[i] += weights[i][j] * gradient[j];
        }
    }

    return input_gradient;
}

void DenseLayer::update(float learning_rate) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[0].size(); ++j) {
            weights[i][j] -= learning_rate * weight_gradients[i][j];
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < biases.size(); ++i) {
        biases[i] -= learning_rate * bias_gradients[i];
    }
}

} // namespace nn
