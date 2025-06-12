# C++ Coding Style Guide

## General Guidelines

1. Use C++20 features when appropriate
2. Follow the principle of RAII (Resource Acquisition Is Initialization)
3. Prefer smart pointers over raw pointers
4. Use const correctness
5. Follow the rule of zero/five

## Naming Conventions

### Files
- Header files: `.hpp`
- Source files: `.cpp`
- Use meaningful, descriptive names
- One class per file unless tightly coupled

### Classes and Structs
- PascalCase: `class NeuralNetwork`
- One class declaration per header file
- Use structs for passive data structures

### Functions
- camelCase: `computeGradient()`
- Verb or verb phrase names
- Clear and descriptive

### Variables
- camelCase: `learningRate`
- Clear, descriptive names
- No Hungarian notation
- Private member variables: `m_` prefix or `_` suffix

### Constants
- ALL_CAPS with underscores
- Meaningful names

### Namespaces
- lowercase: `namespace nn`
- Short, descriptive names

## Formatting

### Indentation
- 4 spaces, no tabs
- Align similar statements

### Line Length
- Maximum 100 characters
- Break long lines at logical points

### Braces
```cpp
if (condition) {
    // code
} else {
    // code
}
```

### Spacing
- One space after keywords
- No space after function names
- Spaces around operators
- One space after commas

## Comments

### General
- Use `//` for single-line comments
- Use `/* */` for multi-line comments
- Write comments in English

### Documentation
- Use Doxygen-style comments for public APIs
- Include parameter descriptions
- Document exceptions
- Include usage examples for complex interfaces

## Best Practices

### Memory Management
- Use smart pointers
- Follow RAII principles
- Avoid manual memory management

### Error Handling
- Use exceptions for error handling
- Document exceptions in function comments
- Use std::optional when appropriate

### Performance
- Use OpenMP for parallelization
- Profile before optimizing
- Document performance considerations

### Templates
- Use when appropriate for generic programming
- Document template parameters
- Consider providing type aliases

## Example

```cpp
namespace nn {

/**
 * @brief Neural network layer interface
 * 
 * Base class for all neural network layers providing
 * forward and backward propagation interfaces.
 */
class Layer {
public:
    virtual ~Layer() = default;
    
    /**
     * @brief Forward pass computation
     * @param input Input tensor
     * @return Output tensor
     */
    virtual std::vector<float> forward(const std::vector<float>& input) = 0;
    
    /**
     * @brief Backward pass computation
     * @param gradient Gradient tensor
     * @return Input gradient tensor
     */
    virtual std::vector<float> backward(const std::vector<float>& gradient) = 0;
    
private:
    std::vector<float> m_cache;
};

} // namespace nn
```

## Version Control

### Commit Messages
- Use present tense ("Add feature" not "Added feature")
- First line is a summary
- Followed by detailed description if needed
- Reference issues and pull requests

### Branches
- `master` for stable releases
- `develop` for development
- Feature branches: `feature/description`
- Bugfix branches: `bugfix/description`

## Testing

### Unit Tests
- Write tests for new features
- Test edge cases
- Name tests descriptively
- One assertion per test

### Integration Tests
- Test component interactions
- Test realistic use cases
- Include performance tests
