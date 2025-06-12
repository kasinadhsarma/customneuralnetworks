#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include <random>
#include <fstream>

namespace nn {

class Layer {
public:
    virtual ~Layer() = default;
    virtual std::vector<float> forward(const std::vector<float>& input) = 0;
    virtual std::vector<float> backward(const std::vector<float>& gradient) = 0;
    virtual void update(float learning_rate) = 0;
};

class DenseLayer : public Layer {
private:
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    std::vector<float> input_cache;
    std::vector<std::vector<float>> weight_gradients;
    std::vector<float> bias_gradients;

public:
    DenseLayer(size_t input_size, size_t output_size);
    std::vector<float> forward(const std::vector<float>& input) override;
    std::vector<float> backward(const std::vector<float>& gradient) override;
    void update(float learning_rate) override;
    
    // Getters and setters for model saving/loading
    const std::vector<std::vector<float>>& get_weights() const { return weights; }
    const std::vector<float>& get_biases() const { return biases; }
    void set_weights(const std::vector<std::vector<float>>& w) { weights = w; }
    void set_biases(const std::vector<float>& b) { biases = b; }
};

class ReLU : public Layer {
private:
    std::vector<float> input_cache;

public:
    float activate(float x);
    float derivative(float x);
    std::vector<float> forward(const std::vector<float>& input) override;
    std::vector<float> backward(const std::vector<float>& gradient) override;
    void update(float learning_rate) override;
};

class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    float learning_rate;
    std::vector<std::vector<float>> layer_outputs;  // Store intermediate outputs
    
    float compute_loss(const std::vector<float>& output, const std::vector<float>& target);

public:
    Model(float lr = 0.01);
    void add_layer(std::unique_ptr<Layer> layer);
    std::vector<float> forward(const std::vector<float>& input);
    void backward(const std::vector<float>& gradient);
    
    // Enhanced training interface
    void train(const std::vector<std::vector<float>>& inputs,
              const std::vector<std::vector<float>>& targets,
              size_t epochs,
              size_t batch_size);
              
    // Model persistence
    void save(const std::string& filename);
    void load(const std::string& filename);
    
    // Prediction and evaluation
    std::vector<float> predict(const std::vector<float>& input);
    float evaluate(const std::vector<std::vector<float>>& inputs,
                  const std::vector<std::vector<float>>& targets);
};

} // namespace nn

#endif // NEURAL_NETWORK_HPP
