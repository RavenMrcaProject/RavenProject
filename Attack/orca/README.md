# ORCA Attacks

## Table of Contents
- [Installation](#installation)
  - [RVO Library](#rvo-library)
  - [Attack Library](#attack-library)
  - [Python Integration](#python-integration)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running Different Cases](#running-different-cases)
  - [Visualization](#visualization)

## Installation

### RVO Library

1. Navigate to the RVO2 directory:
   ```bash
   cd ./RVO2
   ```

2. Create and enter the build directory:
   ```bash
   cd ./build
   ```

3. Generate build files and compile:
   ```bash
   cmake ..
   make
   ```

### Attack Library

1. Navigate to the Attack directory:
   ```bash
   cd ./Attack
   ```

2. Create and enter the build directory:
   ```bash
   cd ./build
   ```

3. Generate build files and compile:
   ```bash
   cmake ..
   make
   ```

### Python Integration

1. Create the shared object (.so) file for Python integration:
   ```bash
   python setup.py build_ext --inplace
   ```

2. Verify the successful creation of the Python module:
   ```python
   python
   >>> import orca_module
   ```

   If no errors appear, the module has been successfully integrated.

## Configuration

Modify the environment settings:
```bash
nano config.txt
```

## Usage

### Running Different Cases

1. Run the benign case:
   ```bash
   python wrapper.py
   ```

2. Run Raven:
   ```bash
   python sample2.py
   ```

### Visualization

To visualize the results:
```bash
python visualize.py
```

## Contributing

Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting

If you encounter any issues during the installation or running process, please check:
- All dependencies are properly installed
- Build directories exist and have write permissions
- Python environment is properly configured
- All paths in config.txt are correctly set