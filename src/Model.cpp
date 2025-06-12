#include "Model.hpp"
#include "DenseLayer.hpp"
#include <stdexcept>
#include <fstream>
#include <nlohmann/json.hpp>

namespace nn {

Model::Model(float learningRate) : learningRate(learningRate) {}

void Model::add_layer(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

std::vector<float> Model::forward(const std::vector<float>& input) {
    if (layers.empty()) {
        throw std::runtime_error("No layers in the model");
    }

    layerOutputs.clear();
    layerOutputs.reserve(layers.size() + 1);
    layerOutputs.push_back(input);

    std::vector<float> current = input;
    for (const auto& layer : layers) {
        current = layer->forward(current);
        layerOutputs.push_back(current);
    }

    return current;
}

std::vector<float> Model::backward(const std::vector<float>& target, const std::vector<float>& output) {
    if (target.size() != output.size()) {
        throw std::invalid_argument("Target and output size mismatch");
    }

    // Calculate initial gradient (assuming MSE loss)
    std::vector<float> gradient(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        gradient[i] = 2.0f * (output[i] - target[i]);
    }

    // Backpropagate through layers
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        gradient = layers[i]->backward(gradient);
    }

    return gradient;
}

void Model::update_parameters() {
    for (auto& layer : layers) {
        layer->updateParameters(learningRate);
    }
}

void Model::train(const std::vector<std::vector<float>>& inputs,
                 const std::vector<std::vector<float>>& targets,
                 size_t epochs,
                 size_t batchSize) {
    if (inputs.empty() || targets.empty() || inputs.size() != targets.size()) {
        throw std::invalid_argument("Invalid input/target data");
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); i += batchSize) {
            size_t batchEnd = std::min(i + batchSize, inputs.size());

            // Process each sample in the batch
            for (size_t j = i; j < batchEnd; ++j) {
                auto output = forward(inputs[j]);
                backward(targets[j], output);
                update_parameters();
            }
        }
    }
}

void Model::save(const std::string& filename) {
    nlohmann::json json;
    json["learning_rate"] = learningRate;
    json["layers"] = nlohmann::json::array();

    for (const auto& layer : layers) {
        nlohmann::json layerJson;
        layer->save(layerJson);
        json["layers"].push_back(layerJson);
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    file << json.dump(4);
}

void Model::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    nlohmann::json json;
    file >> json;

    learningRate = json["learning_rate"];
    layers.clear();

    for (const auto& layerJson : json["layers"]) {
        // Create appropriate layer based on type and load its parameters
        // This part needs to be implemented based on your layer factory pattern
        if (layerJson["type"] == "DenseLayer") {
            auto layer = std::make_unique<DenseLayer>(
                layerJson["input_size"], 
                layerJson["output_size"]
            );
            layer->load(layerJson);
            layers.push_back(std::move(layer));
        }
        // Add other layer types as needed
    }
}

} // namespace nn
