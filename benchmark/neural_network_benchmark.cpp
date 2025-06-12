#include <benchmark/benchmark.h>
#include <random>
#include <memory>
#include "Layer.hpp"
#include "DenseLayer.hpp"
#include "ReLU.hpp"
#include "Model.hpp"
#include <fstream>

namespace nn {
namespace benchmark {

// Utility function to generate random data with bounds checking
inline std::vector<float> generate_random_data(size_t size) {
    if (size == 0 || size > 1000000) { // Add reasonable bounds
        throw std::invalid_argument("Invalid data size");
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> data(size);
    for (auto& val : data) {
        val = dist(gen);
    }
    return data;
}

// Dense Layer Forward Pass Benchmark
static void BM_DenseLayerForward(::benchmark::State& state) {
    const size_t input_size = state.range(0);
    const size_t output_size = state.range(1);
    
    try {
        // Setup
        auto layer = std::make_unique<nn::DenseLayer>(input_size, output_size);
        auto input = generate_random_data(input_size);
        
        // Benchmark loop
        for (auto _ : state) {
            auto output = layer->forward(input);
            ::benchmark::DoNotOptimize(output);
            ::benchmark::ClobberMemory(); // Prevent compiler optimizations
        }
        
        state.SetComplexityN(input_size * output_size);
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// ReLU Activation Benchmark
static void BM_ReLUForward(::benchmark::State& state) {
    const size_t size = state.range(0);
    
    try {
        // Setup
        auto relu = std::make_unique<nn::ReLU>();
        auto input = generate_random_data(size);
        
        // Benchmark loop
        for (auto _ : state) {
            auto output = relu->forward(input);
            ::benchmark::DoNotOptimize(output);
            ::benchmark::ClobberMemory();
        }
        
        state.SetComplexityN(size);
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

// Model Training Benchmark
static void BM_ModelTraining(::benchmark::State& state) {
    const size_t batch_size = state.range(0);
    const size_t input_size = 784;  // MNIST-like input size
    const size_t hidden_size = 128;
    const size_t output_size = 10;
    
    try {
        // Setup model with RAII
        auto model = std::make_unique<nn::Model>(0.01f);
        model->add_layer(std::make_unique<nn::DenseLayer>(input_size, hidden_size));
        model->add_layer(std::make_unique<nn::ReLU>());
        model->add_layer(std::make_unique<nn::DenseLayer>(hidden_size, output_size));
        
        // Generate random training data with bounds checking
        std::vector<std::vector<float>> inputs;
        std::vector<std::vector<float>> targets;
        inputs.reserve(batch_size);
        targets.reserve(batch_size);
        
        for (size_t i = 0; i < batch_size; ++i) {
            inputs.push_back(generate_random_data(input_size));
            targets.push_back(generate_random_data(output_size));
        }
        
        // Benchmark loop
        for (auto _ : state) {
            model->train(inputs, targets, 1, batch_size);
            ::benchmark::ClobberMemory();
        }
        
        state.SetComplexityN(batch_size);
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}

} // namespace benchmark
} // namespace nn

class JSONReporter : public ::benchmark::BenchmarkReporter {
public:
    JSONReporter(const std::string& output_file) : output_file_(output_file) {}

    bool ReportContext(const Context& context) override {
        std::ofstream file(output_file_);
        file << "{\n  \"context\": {\n";
        file << "    \"date\": \"" << context.ReportDate << "\",\n";
        file << "    \"num_cpus\": " << context.num_cpus << ",\n";
        file << "    \"cpu_scaling_enabled\": " << context.cpu_scaling_enabled << "\n  },\n";
        file << "  \"benchmarks\": [\n";
        return true;
    }

    void ReportRuns(const std::vector<Run>& reports) override {
        std::ofstream file(output_file_, std::ios_base::app);
        bool first = true;
        for (const auto& run : reports) {
            if (!first) file << ",\n";
            first = false;
            file << "    {\n";
            file << "      \"name\": \"" << run.run_name << "\",\n";
            file << "      \"iterations\": " << run.iterations << ",\n";
            file << "      \"real_time\": " << run.GetAdjustedRealTime() << ",\n";
            file << "      \"cpu_time\": " << run.GetAdjustedCPUTime() << "\n    }";
        }
        file << "\n  ]\n}\n";
    }

private:
    std::string output_file_;
};

BENCHMARK(nn::benchmark::BM_DenseLayerForward)
    ->Args({64, 32})
    ->Args({128, 64})
    ->Args({256, 128})
    ->Complexity();

BENCHMARK(nn::benchmark::BM_ReLUForward)
    ->Range(8, 1024)
    ->Complexity();

BENCHMARK(nn::benchmark::BM_ModelTraining)
    ->Range(8, 128)
    ->Complexity();

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    
    // Create and register the JSON reporter
    auto json_reporter = std::make_unique<JSONReporter>("benchmark_result.json");
    ::benchmark::RegisterReporter("json", json_reporter.get());
    
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
