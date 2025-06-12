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
};

class Activation : public Layer {
public:
    virtual float activate(float x) = 0;
    virtual float derivative(float x) = 0;
};

class ReLU : public Activation {
private:
    std::vector<float> input_cache;

public:
    float activate(float x) override;
    float derivative(float x) override;
    std::vector<float> forward(const std::vector<float>& input) override;
    std::vector<float> backward(const std::vector<float>& gradient) override;
    void update(float learning_rate) override;
};

class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    float learning_rate;

public:
    Model(float lr = 0.01);
    void add_layer(std::unique_ptr<Layer> layer);
    std::vector<float> forward(const std::vector<float>& input);
    void backward(const std::vector<float>& gradient);
    void train(const std::vector<std::vector<float>>& inputs,
              const std::vector<std::vector<float>>& targets,
              size_t epochs,
              size_t batch_size);
    void save(const std::string& filename);
    void load(const std::string& filename);
};

class Optimizer {
protected:
    float learning_rate;

public:
    explicit Optimizer(float lr) : learning_rate(lr) {}
    virtual ~Optimizer() = default;
    virtual void update(std::vector<float>& params, const std::vector<float>& grads) = 0;
};

class Adam : public Optimizer {
private:
    float beta1;
    float beta2;
    float epsilon;
    std::vector<float> m;
    std::vector<float> v;
    size_t t;

public:
    Adam(float lr = 0.001, float b1 = 0.9, float b2 = 0.999, float eps = 1e-8);
    void update(std::vector<float>& params, const std::vector<float>& grads) override;
};

} // namespace nn

#endif // NEURAL_NETWORK_HPP
