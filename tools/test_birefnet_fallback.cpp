// Standalone regression test; link with the application's Qt/OpenCV/ORT libraries.
#include "src/segmentationsession.cpp"

namespace {
void check(bool condition) {
  if (!condition)
    throw std::runtime_error("BiRefNet fallback regression");
}

class StubSession final : public SegmentationSession {
public:
  explicit StubSession(bool fail, OrtErrorCode code = ORT_EP_FAIL)
      : fail(fail), code(code) {}
  ~StubSession() override { --live; }
  QString displayName() const override { return "BiRefNet"; }
  QImage predictMask(const QImage &) override {
    if (fail)
      throw Ort::Exception("inference failed", code);
    return QImage(1, 1, QImage::Format_Grayscale8);
  }
  inline static int live = 0;
private:
  bool fail;
  OrtErrorCode code;
};

void testFallback(bool failLoading, bool failRunning, bool failLegacy) {
  std::vector<bool> loads;
  CompatibleBiRefNetSession session(
      [&](bool legacy) -> std::unique_ptr<SegmentationSession> {
        loads.push_back(legacy);
        check(StubSession::live == 0); // Failed session released before retry.
        if ((!legacy && failLoading) || (legacy && failLegacy))
          throw Ort::Exception("OpenVINO unsupported operator", ORT_FAIL);
        auto result = std::make_unique<StubSession>(!legacy && failRunning);
        ++StubSession::live;
        return result;
      }, {});
  try {
    check(!session.predictMask({}).isNull());
    check(!session.predictMask({}).isNull());
    check(!failLegacy);
  } catch (const Ort::Exception &) {
    check(failLegacy);
  }
  check(loads == ((failLoading || failRunning) ? std::vector<bool>{false, true}
                                             : std::vector<bool>{false}));
}
}

int main() {
  const char *allocationError =
      "Non-zero status code returned while running Concat node. "
      "BFCArena::AllocateRawInternal Failed to allocate memory for requested "
      "buffer of size 536870912";
  check(isMemoryAllocationError(Ort::Exception(allocationError, ORT_FAIL)));
  check(!isProviderCompatibilityError(Ort::Exception("CUDA out of memory", ORT_EP_FAIL)));
  for (bool gpu : {false, true}) {
    int attempts = 0;
    int fallbacks = 0;
    bool threw = false;
    try {
      const int result = retryOnGpuMemoryError(gpu, [&]() {
        if (++attempts == 1)
          throw Ort::Exception(allocationError, ORT_FAIL);
        return 42;
      }, [&]() { ++fallbacks; });
      check(result == 42);
    } catch (const Ort::Exception &) {
      threw = true;
    }
    check(threw == !gpu);
    check(attempts == (gpu ? 2 : 1));
    check(fallbacks == (gpu ? 1 : 0));
  }
  for (bool memoryError : {false, true}) {
    int attempts = 0;
    int fallbacks = 0;
    bool threw = false;
    try {
      retryOnGpuMemoryError(true, [&]() -> int {
        ++attempts;
        throw Ort::Exception(memoryError ? allocationError : "invalid input",
                             ORT_FAIL);
      }, [&]() { ++fallbacks; });
    } catch (const Ort::Exception &) {
      threw = true;
    }
    check(threw);
    check(attempts == (memoryError ? 2 : 1));
    check(fallbacks == (memoryError ? 1 : 0));
  }
  testFallback(false, false, false);
  testFallback(true, false, false);
  testFallback(false, true, false);
  testFallback(false, true, true);
  check(isProviderCompatibilityError(Ort::Exception("unsupported", ORT_NOT_IMPLEMENTED)));
  check(isProviderCompatibilityError(Ort::Exception("OpenVINO compilation failed", ORT_RUNTIME_EXCEPTION)));
  check(!isProviderCompatibilityError(Ort::Exception("unrelated runtime error", ORT_RUNTIME_EXCEPTION)));
  check(!isProviderCompatibilityError(Ort::Exception("bad model", ORT_INVALID_GRAPH)));
  check(!isProviderCompatibilityError(Ort::Exception("unrelated failure", ORT_FAIL)));
  for (bool ortError : {false, true}) {
    int loads = 0;
    bool threw = false;
    try {
      CompatibleBiRefNetSession session(
          [&](bool legacy) -> std::unique_ptr<SegmentationSession> {
            check(!legacy);
            ++loads;
            if (ortError)
              throw Ort::Exception("invalid input", ORT_INVALID_ARGUMENT);
            throw std::runtime_error("download/hash failure");
          }, {});
    } catch (const std::exception &) {
      threw = true;
    }
    check(threw && loads == 1);
  }
  int loads = 0;
  {
    CompatibleBiRefNetSession invalidInput(
        [&](bool legacy) -> std::unique_ptr<SegmentationSession> {
          check(!legacy);
          ++loads;
          auto result = std::make_unique<StubSession>(true, ORT_INVALID_ARGUMENT);
          ++StubSession::live;
          return result;
        }, {});
    bool threw = false;
    try {
      invalidInput.predictMask({});
    } catch (const Ort::Exception &) {
      threw = true;
    }
    check(threw && loads == 1);
  }
  loads = 0;
  CompatibleBiRefNetSession failingLegacy(
      [&](bool legacy) -> std::unique_ptr<SegmentationSession> {
        ++loads;
        if (!legacy)
          throw Ort::Exception("unsupported", ORT_EP_FAIL);
        auto result = std::make_unique<StubSession>(true);
        ++StubSession::live;
        return result;
      }, {});
  for (int i = 0; i < 2; ++i) {
    bool threw = false;
    try {
      failingLegacy.predictMask({});
    } catch (const Ort::Exception &) {
      threw = true;
    }
    check(threw && loads == 2);
  }
}
