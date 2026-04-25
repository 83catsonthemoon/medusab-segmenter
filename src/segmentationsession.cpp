#include "segmentationsession.h"

#include <QDebug>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

cv::Mat imageToRgbMat(const QImage &img, QImage &rgbStorage) {
    rgbStorage = img.convertToFormat(QImage::Format_RGB888);
    return cv::Mat(
        rgbStorage.height(),
        rgbStorage.width(),
        CV_8UC3,
        const_cast<uchar*>(rgbStorage.constBits()),
        static_cast<size_t>(rgbStorage.bytesPerLine())
    );
}

std::vector<float> matToNchwFloat(const cv::Mat &mat) {
    const int w = mat.cols;
    const int h = mat.rows;
    std::vector<float> values(3 * h * w);

    for (int y = 0; y < h; ++y) {
        const uchar *line = mat.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            const int idx = y * w + x;
            values[idx] = line[x * 3 + 0] / 255.0f;
            values[h * w + idx] = line[x * 3 + 1] / 255.0f;
            values[2 * h * w + idx] = line[x * 3 + 2] / 255.0f;
        }
    }

    return values;
}

QImage grayscaleMatToImageCopy(const cv::Mat &mat) {
    QImage image(
        mat.data,
        mat.cols,
        mat.rows,
        static_cast<int>(mat.step),
        QImage::Format_Grayscale8
    );
    return image.copy();
}

class OnnxSegmentationSession : public SegmentationSession {
public:
    OnnxSegmentationSession(Ort::Env &env, const char *modelPath, int fallbackInputSize)
        : inputSize(fallbackInputSize),
          session(env, modelPath, makeSessionOptions()) {
        readInputSize(fallbackInputSize);
    }

protected:
    Ort::SessionOptions makeSessionOptions() {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        return opts;
    }

    void readInputSize(int fallbackInputSize) {
        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> inputShape = inputTensorInfo.GetShape();

        if (inputShape.size() == 4 && inputShape[2] > 0 && inputShape[3] > 0) {
            inputSize = static_cast<int>(inputShape[2]);
        } else {
            inputSize = fallbackInputSize;
        }

        qDebug() << "Model input size:" << inputSize;
    }

    Ort::Value run(const cv::Mat &inputMat, std::vector<float> &inputTensorValues) {
        inputTensorValues = matToNchwFloat(inputMat);

        std::array<int64_t, 4> shape{1, 3, inputMat.rows, inputMat.cols};
        qDebug() << "Running" << displayName() << "with input tensor shape:"
                 << shape[0] << shape[1] << shape[2] << shape[3];

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            mem,
            inputTensorValues.data(),
            inputTensorValues.size(),
            shape.data(),
            shape.size()
        );

        auto inputName = session.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
        auto outputName = session.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());

        std::vector<const char*> inputNames{inputName.get()};
        std::vector<const char*> outputNames{outputName.get()};

        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            &inputTensor,
            1,
            outputNames.data(),
            1
        );

        return std::move(outputs[0]);
    }

    static int outputHeight(const std::vector<int64_t> &shape, int fallback) {
        if (shape.size() >= 2 && shape[shape.size() - 2] > 0)
            return static_cast<int>(shape[shape.size() - 2]);
        return fallback;
    }

    static int outputWidth(const std::vector<int64_t> &shape, int fallback) {
        if (shape.size() >= 1 && shape[shape.size() - 1] > 0)
            return static_cast<int>(shape[shape.size() - 1]);
        return fallback;
    }

    int inputSize;
    Ort::Session session;
};

class U2NetSession final : public OnnxSegmentationSession {
public:
    U2NetSession(Ort::Env &env, const char *modelPath, int fallbackInputSize, QString name)
        : OnnxSegmentationSession(env, modelPath, fallbackInputSize),
          modelName(std::move(name)) {}

    QString displayName() const override {
        return modelName;
    }

    QImage predictMask(const QImage &img) override {
        QImage rgbStorage;
        cv::Mat originalMat = imageToRgbMat(img, rgbStorage);

        cv::Mat inputMat;
        cv::resize(
            originalMat,
            inputMat,
            cv::Size(inputSize, inputSize),
            0.0,
            0.0,
            cv::INTER_LANCZOS4
        );

        std::vector<float> inputTensorValues;
        Ort::Value output = run(inputMat, inputTensorValues);

        float *outData = output.GetTensorMutableData<float>();
        auto outputInfo = output.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> outputShape = outputInfo.GetShape();
        const int maskH = outputHeight(outputShape, inputSize);
        const int maskW = outputWidth(outputShape, inputSize);

        float minValue = std::numeric_limits<float>::max();
        float maxValue = std::numeric_limits<float>::lowest();
        for (int i = 0; i < maskH * maskW; ++i) {
            minValue = std::min(minValue, outData[i]);
            maxValue = std::max(maxValue, outData[i]);
        }

        const float range = maxValue - minValue;
        cv::Mat mask(maskH, maskW, CV_8UC1);
        for (int y = 0; y < maskH; ++y) {
            uchar *line = mask.ptr<uchar>(y);
            for (int x = 0; x < maskW; ++x) {
                const int idx = y * maskW + x;
                const float value = range > 0.0f ? (outData[idx] - minValue) / range : 0.0f;
                line[x] = static_cast<uchar>(std::clamp(value, 0.0f, 1.0f) * 255.0f);
            }
        }

        cv::Mat scaledMask;
        cv::resize(
            mask,
            scaledMask,
            cv::Size(originalMat.cols, originalMat.rows),
            0.0,
            0.0,
            cv::INTER_LANCZOS4
        );

        return grayscaleMatToImageCopy(scaledMask);
    }

private:
    QString modelName;
};

class BiRefNetSession final : public OnnxSegmentationSession {
public:
    explicit BiRefNetSession(Ort::Env &env)
        : OnnxSegmentationSession(env, "models/birefnet.onnx", 1440) {}

    QString displayName() const override {
        return "BiRefNet";
    }

    QImage predictMask(const QImage &img) override {
        QImage rgbStorage;
        cv::Mat originalMat = imageToRgbMat(img, rgbStorage);

        int resizedW = inputSize;
        int resizedH = inputSize;
        if (originalMat.rows >= originalMat.cols) {
            resizedH = inputSize;
            resizedW = std::max(1, static_cast<int>(
                std::lround(static_cast<double>(originalMat.cols) * inputSize / originalMat.rows)
            ));
        } else {
            resizedW = inputSize;
            resizedH = std::max(1, static_cast<int>(
                std::lround(static_cast<double>(originalMat.rows) * inputSize / originalMat.cols)
            ));
        }

        cv::Mat resizedMat;
        cv::resize(
            originalMat,
            resizedMat,
            cv::Size(resizedW, resizedH),
            0.0,
            0.0,
            cv::INTER_LANCZOS4
        );

        cv::Mat inputMat = cv::Mat::zeros(inputSize, inputSize, CV_8UC3);
        resizedMat.copyTo(inputMat(cv::Rect(0, 0, resizedW, resizedH)));

        std::vector<float> inputTensorValues;
        Ort::Value output = run(inputMat, inputTensorValues);

        float *outData = output.GetTensorMutableData<float>();
        auto outputInfo = output.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> outputShape = outputInfo.GetShape();
        const int maskH = outputHeight(outputShape, inputSize);
        const int maskW = outputWidth(outputShape, inputSize);

        cv::Mat mask(maskH, maskW, CV_8UC1);
        for (int y = 0; y < maskH; ++y) {
            uchar *line = mask.ptr<uchar>(y);
            for (int x = 0; x < maskW; ++x) {
                const int idx = y * maskW + x;
                line[x] = static_cast<uchar>(std::clamp(outData[idx], 0.0f, 1.0f) * 255.0f);
            }
        }

        const int croppedW = std::min(resizedW, mask.cols);
        const int croppedH = std::min(resizedH, mask.rows);
        cv::Mat croppedMask = mask(cv::Rect(0, 0, croppedW, croppedH));

        cv::Mat scaledMask;
        cv::resize(
            croppedMask,
            scaledMask,
            cv::Size(originalMat.cols, originalMat.rows),
            0.0,
            0.0,
            cv::INTER_LANCZOS4
        );

        return grayscaleMatToImageCopy(scaledMask);
    }
};

} // namespace

std::unique_ptr<SegmentationSession> createSegmentationSession(
    ModelKind kind,
    Ort::Env &env
) {
    switch (kind) {
    case ModelKind::U2Net:
        return std::make_unique<U2NetSession>(env, "models/u2net.onnx", 1024, "U2Net");
    case ModelKind::BiRefNet:
        return std::make_unique<BiRefNetSession>(env);
    case ModelKind::U2NetPChunks:
        return std::make_unique<U2NetSession>(
            env,
            "models/u2netp_chunks.onnx",
            512,
            "U2NetP Chunks"
        );
    }

    throw std::invalid_argument("Unknown model kind");
}
