#include "DenseLayer.hpp"
#include <cmath>
#include <stdexcept>
#include <random>

namespace nn {

DenseLayer::DenseLayer(size_t inputSize, size_t outputSize)
    : inputSize(inputSize), outputSize(outputSize) {
    // Initialize weights using Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(6.0f / (inputSize + outputSize));
    std::uniform_real_distribution<float> dis(-limit, limit);

    // Initialize weights
    weights.resize(outputSize, std::vector<float>(inputSize));
    for (auto& row : weights) {
        for (float& weight : row) {
            weight = dis(gen);
        }
    }

    // Initialize biases to zero
    biases = std::vector<float>(outputSize, 0.0f);
}

std::vector<float> DenseLayer::forward(const std::vector<float>& input) {
    if (input.size() != inputSize) {
        throw std::invalid_argument("Input size does not match layer input dimension");
    }

    lastInput = input;
    lastOutput.resize(outputSize);

    #pragma omp parallel for
    for (size_t i = 0; i < outputSize; ++i) {
        float sum = biases[i];
        for (size_t j = 0; j < inputSize; ++j) {
            sum += weights[i][j] * input[j];
        }
        lastOutput[i] = sum;
    }

    return lastOutput;
}

std::vector<float> DenseLayer::backward(const std::vector<float>& gradient) {
    if (gradient.size() != outputSize) {
        throw std::invalid_argument("Gradient size does not match layer output dimension");
    }

    std::vector<float> inputGradient(inputSize, 0.0f);

    // Calculate input gradients
    #pragma omp parallel for
    for (size_t i = 0; i < inputSize; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < outputSize; ++j) {
            sum += weights[j][i] * gradient[j];
        }
        inputGradient[i] = sum;
    }

    return inputGradient;
}

void DenseLayer::updateParameters(float learningRate) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < outputSize; ++i) {
        for (size_t j = 0; j < inputSize; ++j) {
            weights[i][j] -= learningRate * lastInput[j];
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < outputSize; ++i) {
        biases[i] -= learningRate;
    }
}

void DenseLayer::save(nlohmann::json& json) const {
    json["type"] = "DenseLayer";
    json["input_size"] = inputSize;
    json["output_size"] = outputSize;
    json["weights"] = weights;
    json["biases"] = biases;
}

void DenseLayer::load(const nlohmann::json& json) {
    if (json["type"] != "DenseLayer") {
        throw std::runtime_error("Invalid layer type in JSON");
    }

    inputSize = json["input_size"];
    outputSize = json["output_size"];
    weights = json["weights"].get<std::vector<std::vector<float>>>();
    biases = json["biases"].get<std::vector<float>>();
}

} // namespace nn
