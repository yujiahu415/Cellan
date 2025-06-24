# Cellan: Analyze cell images.

[![PyPI - Version](https://img.shields.io/pypi/v/Cellan)](https://pypi.org/project/Cellan/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Cellan)](https://pypi.org/project/Cellan/)
[![Downloads](https://static.pepy.tech/badge/Cellan)](https://pepy.tech/project/Cellan)

<p>&nbsp;</p>

## Installation

<p>&nbsp;</p>

Cellan works for Windows, Mac and Linux systems. Installation steps can vary for different systems. But in general, you need to:
1) Install Microsoft Visual studio build tools
2) Install Python3 (3.9 or 3.10)
3) Set up CUDA (v11.8) for GPU usage
4) Install Cellan with pip
5) If using GPU, install PyTorch==2.0.1 with cu118.

Below is the guide for Windows.

1. Install [Git](https://git-scm.com/download/win). 

   Select the `64-bit Git for Windows Setup` option. Run the installer, and accept all default values.

2. Install the [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022). 

   Scroll down to the entry that says `Build Tools for Visual Studio 2022` and click "Download". When you run the downloaded executable, you will be prompted to choose what tools you will need. Select only the `Desktop Development With C++` workload, then click 'Install'.

3. Install [Python 3.10](https://www.python.org/downloads/release/python-31011/).

   Scroll down to the bottom and click the `Windows installer (64-bit)` option. Run the installer and select 'Add python to path' and 'Disable long path limit'.

4. If you're using an NVIDIA GPU, install CUDA Toolkit 11.8 and cuDNN.

   Install [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64). Select your version of Windows, select "exe (local)," then click "Download."

   To verify your installation of CUDA, use the following command.

   ```pwsh-session
   set CUDA_HOME=%CUDA_HOME_V11_8%
   ```
   ```pwsh-session
   nvcc --version
   ```

   Finally, install [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive). You will need to register an Nvidia Developer account, which you can do for free. You can choose cuDNN v8.9.7 that supports CUDA toolkit v11.8. Choose 'Local Installer for Windows (Zip)', download and extract it. And then copy the three folders 'bin', 'lib', and 'include' into where the CUDA toolkit is installed (typcially, 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\'), and replace all the three folders with the same names. After that, you may need to add the 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8' to path via environmental variables.

5. Upgrade `pip`, `wheel`, `setuptools`.
   
   ```pwsh-session
   py -3.10 -m pip install --upgrade pip wheel setuptools
   ```

6. Install Cellan via `pip`.
   
   ```pwsh-session
   py -3.10 -m pip install Cellan
   ```

7. If you're using an NVIDIA GPU, install PyTorch with cu118:
   
   ```pwsh-session
   py -3.10 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
   ```

Launch Cellan:

   ```pwsh-session
   Cellan
   ```

   If this doesn't work, which typically is because the python3/script is not in your environment path. You can google 'add python3 script to path in environmental variable in windows' to add it to path, or simply use the following commands to initiate Cellan:

   ```pwsh-session
   py -3.10
   ```
   ```pwsh-session
   from Cellan import __main__
   ```
   ```pwsh-session
   __main__.main()
   ```

