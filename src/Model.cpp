#include "NeuralNetwork.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <cmath>
#include <random>
#include <algorithm>

namespace nn {

Model::Model(float lr) : learning_rate(lr) {}

void Model::add_layer(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

std::vector<float> Model::forward(const std::vector<float>& input) {
    std::vector<float> current = input;
    layer_outputs.clear();
    layer_outputs.push_back(current);

    for (auto& layer : layers) {
        current = layer->forward(current);
        layer_outputs.push_back(current);
    }
    return current;
}

void Model::backward(const std::vector<float>& gradient) {
    std::vector<float> current = gradient;
    for (int i = layers.size() - 1; i >= 0; --i) {
        current = layers[i]->backward(current);
    }
}

float Model::compute_loss(const std::vector<float>& output, const std::vector<float>& target) {
    float loss = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        float diff = output[i] - target[i];
        loss += diff * diff;
    }
    return loss / output.size();
}

void Model::train(const std::vector<std::vector<float>>& inputs,
                 const std::vector<std::vector<float>>& targets,
                 size_t epochs,
                 size_t batch_size) {
    
    if (inputs.empty() || targets.empty() || inputs.size() != targets.size()) {
        throw std::runtime_error("Invalid input/target data");
    }

    size_t num_samples = inputs.size();
    std::vector<size_t> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 rng(rd());

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        
        // Shuffle training data using modern C++ random facilities
        std::shuffle(indices.begin(), indices.end(), rng);
        
        // Mini-batch training
        for (size_t i = 0; i < num_samples; i += batch_size) {
            size_t current_batch_size = std::min(batch_size, num_samples - i);
            
            // Accumulate gradients over mini-batch
            std::vector<float> batch_gradient;
            float batch_loss = 0.0f;
            
            #pragma omp parallel for reduction(+:batch_loss)
            for (size_t j = 0; j < current_batch_size; ++j) {
                size_t idx = indices[i + j];
                auto output = forward(inputs[idx]);
                
                // Compute loss and gradients
                std::vector<float> gradients(output.size());
                for (size_t k = 0; k < output.size(); ++k) {
                    float diff = output[k] - targets[idx][k];
                    gradients[k] = 2.0f * diff / output.size();
                    batch_loss += diff * diff;
                }
                
                // Backward pass
                backward(gradients);
                
                // Update parameters
                for (auto& layer : layers) {
                    layer->update(learning_rate / current_batch_size);
                }
            }
            
            total_loss += batch_loss / current_batch_size;
        }
        
        // Print training progress
        float avg_loss = total_loss / (num_samples / batch_size);
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << ", Average Loss: " << avg_loss << std::endl;
        }
        
        // Early stopping if loss is small enough
        if (avg_loss < 1e-6) {
            std::cout << "Converged at epoch " << epoch + 1 << std::endl;
            break;
        }
    }
}

void Model::save(const std::string& filename) {
    nlohmann::json model_json;
    
    // Save model hyperparameters
    model_json["learning_rate"] = learning_rate;
    model_json["num_layers"] = layers.size();
    
    // Save layers
    std::vector<nlohmann::json> layers_json;
    for (size_t i = 0; i < layers.size(); ++i) {
        nlohmann::json layer_json;
        
        // Save layer type
        if (auto* dense = dynamic_cast<DenseLayer*>(layers[i].get())) {
            layer_json["type"] = "dense";
            layer_json["weights"] = dense->get_weights();
            layer_json["biases"] = dense->get_biases();
        } else if (dynamic_cast<ReLU*>(layers[i].get())) {
            layer_json["type"] = "relu";
        }
        
        layers_json.push_back(layer_json);
    }
    model_json["layers"] = layers_json;
    
    // Write to file
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for saving model: " + filename);
    }
    file << model_json.dump(4);
}

void Model::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for loading model: " + filename);
    }
    
    nlohmann::json model_json;
    file >> model_json;
    
    // Load model hyperparameters
    learning_rate = model_json["learning_rate"];
    
    // Clear existing layers
    layers.clear();
    
    // Load layers
    for (const auto& layer_json : model_json["layers"]) {
        std::string layer_type = layer_json["type"];
        
        if (layer_type == "dense") {
            auto weights = layer_json["weights"].get<std::vector<std::vector<float>>>();
            auto biases = layer_json["biases"].get<std::vector<float>>();
            
            auto dense = std::make_unique<DenseLayer>(weights[0].size(), biases.size());
            dense->set_weights(weights);
            dense->set_biases(biases);
            layers.push_back(std::move(dense));
        }
        else if (layer_type == "relu") {
            layers.push_back(std::make_unique<ReLU>());
        }
    }
}

std::vector<float> Model::predict(const std::vector<float>& input) {
    return forward(input);
}

float Model::evaluate(const std::vector<std::vector<float>>& inputs,
                     const std::vector<std::vector<float>>& targets) {
    if (inputs.empty() || targets.empty() || inputs.size() != targets.size()) {
        throw std::runtime_error("Invalid input/target data");
    }
    
    float total_loss = 0.0f;
    
    #pragma omp parallel for reduction(+:total_loss)
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = predict(inputs[i]);
        total_loss += compute_loss(output, targets[i]);
    }
    
    return total_loss / inputs.size();
}

} // namespace nn
