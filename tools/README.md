# BiRefNet validation

Enable the optional tools with the normal application dependencies installed:

```bash
cmake -S . -B build -DMEDUSAB_BUILD_VALIDATION=ON
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

The regression test injects model loaders and inference errors to check default
checkpoint selection, fallback during loading and inference, release of the
failed session before loading its replacement, propagation of unrelated errors,
and propagation of legacy failures without another fallback. It does not require
model downloads or a particular execution provider.

It also checks GPU memory fallback using the reported BFCArena allocation error:
one CPU retry after GPU allocation failure, no retry for unrelated errors or
CPU allocation failures, and propagation of errors from the CPU retry. GPU memory
errors do not trigger BiRefNet checkpoint fallback.

On Linux, compare checkpoints sequentially in fresh processes:

```bash
build/bin/benchmark-birefnet-memory /tmp/birefnet_22.onnx
build/bin/benchmark-birefnet-memory "$HOME/.medusab-segmenter-cache/birefnet.onnx"
```

An optional second argument sets the inference count (default: one). The tool
uses CPU execution, four intra-op threads, full graph optimization, default ORT
memory arenas/patterns, the model's static input shape, and identical deterministic
float32 RGB input in [0, 1]. It requests only the first output, like the app.
It measures process peak resident memory using Linux `getrusage`, including
model loading and input/output buffers. A child process isolates the measurements
from the launcher's pre-exec memory high-water mark. These numbers exclude model
download buffers, Qt image processing, and GPU memory.

See [the measured comparison](birefnet-memory-results.md) for release wording and
the limits of the measurement.
