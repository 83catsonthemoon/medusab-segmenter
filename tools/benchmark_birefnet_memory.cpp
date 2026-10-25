// Linux CPU benchmark: run each checkpoint in a fresh process.
// Uses the application's ORT thread count, graph optimization, and input layout.
#include <onnxruntime_cxx_api.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <array>
#include <iostream>
#include <vector>

static long peakKiB() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: benchmark_birefnet_memory checkpoint.onnx [inferences=1]\n";
    return 2;
  }
  // Linux preserves a pre-exec RSS high-water mark. Fork here so compiler or
  // launcher memory cannot contaminate the checkpoint's loading measurement.
  const pid_t child = fork();
  if (child < 0)
    return 1;
  if (child > 0) {
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
      if (errno != EINTR)
        return 1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
  }
  try {
    const int runs = argc == 3 ? std::stoi(argv[2]) : 1;
    if (runs < 1)
      throw std::runtime_error("Inference count must be positive");
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "birefnet-memory");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4);
    options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    Ort::Session session(env, argv[1], options);
    const long loadPeak = peakKiB();
    auto type = session.GetInputTypeInfo(0);
    auto shape = type.GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "input_shape=";
    for (auto dimension : shape)
      std::cout << dimension << ',';
    std::cout << std::endl;
    size_t count = 1;
    for (auto dimension : shape) {
      if (dimension <= 0)
        throw std::runtime_error("Benchmark requires a static input shape");
      count *= static_cast<size_t>(dimension);
    }
    // Identical deterministic RGB input in [0, 1] for both exports.
    std::vector<float> input(count);
    for (size_t i = 0; i < count; ++i)
      input[i] = static_cast<float>(i % 256) / 255.0f;
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(
        memory, input.data(), count, shape.data(), shape.size());
    auto inputName = session.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    auto outputName = session.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    const char *inputs[] = {inputName.get()};
    const char *outputs[] = {outputName.get()};
    for (int i = 0; i < runs; ++i) {
      auto result = session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
      auto info = result[0].GetTensorTypeAndShapeInfo();
      auto data = result[0].GetTensorData<float>();
      double sum = 0;
      for (size_t j = 0; j < info.GetElementCount(); ++j)
        sum += data[j];
      std::cout << "run=" << i + 1 << " output_sum=" << sum
                << " peak_rss_kib=" << peakKiB() << std::endl;
    }
    std::cout << "load_peak_rss_kib=" << loadPeak << std::endl;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
