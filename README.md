# medusab-segmenter

## Development Requirements

Linux:
```bash
build-essential cmake ninja-build qt6-base-dev
```

### Build

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT=onnxruntime-linux-x64-gpu-1.25.0
cmake --build build
```

### ONNX Runtime

Grab the [latest release](https://github.com/microsoft/onnxruntime/releases)
