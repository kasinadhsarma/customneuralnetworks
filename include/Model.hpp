#ifndef CUSTOMNEURALNETWORKS_MODEL_HPP
#define CUSTOMNEURALNETWORKS_MODEL_HPP

#include "Layer.hpp"
#include <memory>
#include <vector>
#include <string>

namespace nn {

class Model {
public:
    explicit Model(float learningRate = 0.01f);
    ~Model() = default;

    // Add a layer to the model
    void add_layer(std::unique_ptr<Layer> layer);

    // Forward pass
    std::vector<float> forward(const std::vector<float>& input);

    // Training
    void train(const std::vector<std::vector<float>>& inputs,
              const std::vector<std::vector<float>>& targets,
              size_t epochs,
              size_t batchSize);

    // Model persistence
    void save(const std::string& filename);
    void load(const std::string& filename);

private:
    std::vector<std::unique_ptr<Layer>> layers;
    float learningRate;
    
    // Helper methods
    std::vector<float> backward(const std::vector<float>& target, const std::vector<float>& output);
    void update_parameters();
    
    // Store intermediate outputs for backpropagation
    std::vector<std::vector<float>> layerOutputs;
};

} // namespace nn

#endif // CUSTOMNEURALNETWORKS_MODEL_HPP
