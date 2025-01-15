# GLAS Attacks

## Table of Contents
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running Different Cases](#running-different-cases)

## Installation

Before you begin, ensure you have all the necessary prerequisites installed on your system.

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** Package versions must be strictly followed to avoid compatibility issues.

2. Activate the virtual environment:
   ```bash
   source ./glas_env/bin/activate
   ```

## Configuration

1. Navigate to the examples directory:
   ```bash
   cd ./code/examples
   ```

2. Configure your environment by modifying `config.txt` according to your needs.

## Usage

### Running Different Cases

1. Execute the benign case:
   ```bash
   python benign_glas_sample.py
   ```

2. Run Raven:
   ```bash
   python sample2.py
   ```

3. Run the visualizer:
   - First, update the configuration in `config.txt`
   - Then execute:
     ```bash
     python visualize_chat40_glas.py
     ```

## Contributing

Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.