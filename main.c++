#include "NeuralNetwork.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// Example XOR problem
std::vector<std::vector<float>> generateXORData(size_t samples) {
    std::vector<std::vector<float>> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise(-0.1f, 0.1f);

    for (size_t i = 0; i < samples; ++i) {
        bool x1 = gen() % 2;
        bool x2 = gen() % 2;
        bool y = x1 ^ x2;
        data.push_back({static_cast<float>(x1) + noise(gen), 
                       static_cast<float>(x2) + noise(gen), 
                       static_cast<float>(y)});
    }
    return data;
}

int main() {
    // Create training data
    const size_t NUM_SAMPLES = 1000;
    auto data = generateXORData(NUM_SAMPLES);
    
    std::vector<std::vector<float>> inputs(NUM_SAMPLES, std::vector<float>(2));
    std::vector<std::vector<float>> targets(NUM_SAMPLES, std::vector<float>(1));
    
    for (size_t i = 0; i < NUM_SAMPLES; ++i) {
        inputs[i] = {data[i][0], data[i][1]};
        targets[i] = {data[i][2]};
    }

    // Create model
    nn::Model model(0.01f);
    
    // Add layers
    model.add_layer(std::make_unique<nn::DenseLayer>(2, 8));
    model.add_layer(std::make_unique<nn::ReLU>());
    model.add_layer(std::make_unique<nn::DenseLayer>(8, 8));
    model.add_layer(std::make_unique<nn::ReLU>());
    model.add_layer(std::make_unique<nn::DenseLayer>(8, 1));

    // Train model
    std::cout << "Training XOR neural network...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    model.train(inputs, targets, 100, 32);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training completed in " << duration.count() << "ms\n\n";

    // Test the model
    std::cout << "Testing the model:\n";
    std::vector<std::vector<float>> test_inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    
    for (const auto& input : test_inputs) {
        auto output = model.forward(input);
        std::cout << input[0] << " XOR " << input[1] << " = " << output[0] << "\n";
    }

    return 0;
}