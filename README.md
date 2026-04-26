# medusab-segmenter

Because skin segmentation is laborious.

Grab the latest release from the [releases](https://github.com/83catsonthemoon/medusab-segmenter/releases) tab

![demo](./assets/demo.png)

## Development Requirements

### Requirements

Linux:
```bash
build-essential cmake ninja-build qt6-base-dev
```

Windows
```ps
choco install qt6-base-dev
```

ONNX Runtime:

Grab the [latest release](https://github.com/microsoft/onnxruntime/releases) for your platform.

#### CUDA

If you're on Linux, make sure you've got the typical CUDA drivers and such if you are using an NVIDIA GPU, including `libcudnn`.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install libcudnn9-cuda-12
sudo ldconfig
```

For Windows, go download the dev kit as linked in the ONNX documentation.

### Build

#### Linux

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT=onnxruntime-linux-x64-gpu-1.25.0
cmake --build build
```

#### Windows 

MinGW
```ps
cmake -S . -B build -G "MinGW Makefiles" `
   -DCMAKE_C_COMPILER=gcc `
   -DCMAKE_CXX_COMPILER=g++ `
   -DCMAKE_PREFIX_PATH="C:/Qt/6.4.2/mingw_64" `
   -DONNXRUNTIME_ROOT="onnxruntime-win-x64-gpu-1.25.0"
```

Maybe MSVC
```ps
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_PREFIX_PATH="C:/Qt/6.4.2/msvc2019_64" `
  -DONNXRUNTIME_ROOT="onnxruntime-win-x64-gpu-1.25.0"
```
