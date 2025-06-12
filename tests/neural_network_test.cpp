#include <gtest/gtest.h>
#include "../include/NeuralNetwork.hpp"
#include "../include/DenseLayer.hpp"
#include "../include/ReLU.hpp"

TEST(NeuralNetworkTest, LayerCreation) {
    DenseLayer layer(2, 3);  // 2 inputs, 3 outputs
    EXPECT_EQ(layer.getInputSize(), 2);
    EXPECT_EQ(layer.getOutputSize(), 3);
}

TEST(NeuralNetworkTest, ReLUActivation) {
    ReLU relu;
    std::vector<double> input = {-1.0, 0.0, 1.0};
    std::vector<double> output = relu.forward(input);
    
    EXPECT_EQ(output.size(), 3);
    EXPECT_DOUBLE_EQ(output[0], 0.0);  // ReLU of -1 should be 0
    EXPECT_DOUBLE_EQ(output[1], 0.0);  // ReLU of 0 should be 0
    EXPECT_DOUBLE_EQ(output[2], 1.0);  // ReLU of 1 should be 1
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
