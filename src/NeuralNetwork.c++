#include "NeuralNetwork.hpp"
#include <iostream>
#include <vector>
#include <chrono>
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
        output[i] = activate(input[i]);
    }

    return output;
}

std::vector<float> ReLU::backward(const std::vector<float>& gradient) {
    std::vector<float> output(gradient.size());

    #pragma omp parallel for
    for (size_t i = 0; i < gradient.size(); ++i) {
        output[i] = gradient[i] * derivative(input_cache[i]);
    }

    return output;
}

void ReLU::update(float learning_rate) {
    // ReLU has no parameters to update
}

Model::Model(float lr) : learning_rate(lr) {}

void Model::add_layer(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

std::vector<float> Model::forward(const std::vector<float>& input) {
    std::vector<float> current = input;
    for (auto& layer : layers) {
        current = layer->forward(current);
    }
    return current;
}

void Model::backward(const std::vector<float>& gradient) {
    std::vector<float> current = gradient;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        current = (*it)->backward(current);
    }
}

void Model::train(const std::vector<std::vector<float>>& inputs,
                 const std::vector<std::vector<float>>& targets,
                 size_t epochs,
                 size_t batch_size) {
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < inputs.size(); i += batch_size) {
            size_t current_batch_size = std::min(batch_size, inputs.size() - i);
            
            // Forward pass
            std::vector<std::vector<float>> batch_outputs;
            for (size_t j = 0; j < current_batch_size; ++j) {
                batch_outputs.push_back(forward(inputs[i + j]));
            }
            
            // Compute gradients
            std::vector<float> batch_gradient(batch_outputs[0].size(), 0.0f);
            for (size_t j = 0; j < current_batch_size; ++j) {
                for (size_t k = 0; k < batch_outputs[j].size(); ++k) {
                    float diff = batch_outputs[j][k] - targets[i + j][k];
                    batch_gradient[k] += diff / current_batch_size;
                    total_loss += diff * diff;
                }
            }
            
            // Backward pass
            backward(batch_gradient);
            
            // Update weights
            for (auto& layer : layers) {
                layer->update(learning_rate);
            }
        }
        
        std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                  << ", Loss: " << total_loss / inputs.size() << std::endl;
    }
}

void Model::save(const std::string& filename) {
    // Implementation for model saving
}

void Model::load(const std::string& filename) {
    // Implementation for model loading
}

} // namespace nn
