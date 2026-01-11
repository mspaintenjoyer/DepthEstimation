# Python Environment Setup

Follow these steps to prepare the project environment:

1. **Create a virtual environment**
    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment**
    - Windows
      ```bash
      venv\Scripts\activate
      ```
    - macOS/Linux
      ```bash
      source venv/bin/activate
      ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    - CPU-only PyTorch
      ```bash
      pip install torch
      ```
    - CUDA PyTorch (pick the right CUDA wheel from PyTorch site)
      ```bash
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
      ```
    See https://pytorch.org/get-started/locally/ to choose the correct CUDA version.