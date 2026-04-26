#include "segmentationsession.h"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDebug>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QSaveFile>
#include <QStandardPaths>
#include <QUrl>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct TileBox {
    int x0;
    int y0;
    int x1;
    int y1;
};

constexpr int ChunkTileSize = 512;
constexpr int ChunkOverlap = 64;
constexpr float ChunkMinPosFrac = 0.01f;
constexpr int ChunkBinThresh = 0;
constexpr int ChunkEdgeBandPx = 12;
constexpr float ChunkEdgeTargetFrac = 0.8f;
constexpr float ChunkPosPureFrac = 0.1f;
constexpr float ChunkNegPureFrac = 0.1f;
constexpr const char *CacheDirName = ".medusab-segmenter-cache";

QString sha256ForFile(const QString &path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
        throw std::runtime_error(("Failed to open file for hashing: " + path).toStdString());

    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!hash.addData(&file))
        throw std::runtime_error(("Failed to hash file: " + path).toStdString());

    return hash.result().toHex();
}

bool fileMatchesHash(const QString &path, const QString &expectedSha256) {
    return QFile::exists(path) && sha256ForFile(path).compare(expectedSha256, Qt::CaseInsensitive) == 0;
}

QString cacheDirectoryPath() {
    const QString homeDir = QStandardPaths::writableLocation(QStandardPaths::HomeLocation);
    return QDir(homeDir).filePath(CacheDirName);
}

QString ensureModelAvailable(
    const QString &fileName,
    const QString &downloadUrl,
    const QString &expectedSha256,
    const std::function<void(const QString &)> &statusCallback
) {
    const QString cacheDirPath = cacheDirectoryPath();
    QDir cacheDir(cacheDirPath);
    if (!cacheDir.exists() && !cacheDir.mkpath(".")) {
        throw std::runtime_error(("Failed to create cache directory: " + cacheDirPath).toStdString());
    }

    const QString cachedModelPath = cacheDir.filePath(fileName);
    if (fileMatchesHash(cachedModelPath, expectedSha256))
        return cachedModelPath;

    if (QFile::exists(cachedModelPath) && !QFile::remove(cachedModelPath)) {
        throw std::runtime_error(("Cached model hash mismatch and file could not be removed: " + cachedModelPath).toStdString());
    }

    if (statusCallback)
        statusCallback(QString("Downloading model..."));

    QNetworkAccessManager networkManager;
    QNetworkRequest request{QUrl(downloadUrl)};
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);

    QEventLoop loop;
    QNetworkReply *reply = networkManager.get(request);
    QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    loop.exec();

    std::unique_ptr<QNetworkReply, void(*)(QNetworkReply*)> replyGuard(reply, [](QNetworkReply *r) {
        if (r)
            r->deleteLater();
    });

    if (reply->error() != QNetworkReply::NoError) {
        throw std::runtime_error(("Failed to download model: " + reply->errorString()).toStdString());
    }

    QSaveFile outFile(cachedModelPath);
    if (!outFile.open(QIODevice::WriteOnly)) {
        throw std::runtime_error(("Failed to write cached model: " + cachedModelPath).toStdString());
    }

    outFile.write(reply->readAll());
    if (!outFile.commit()) {
        throw std::runtime_error(("Failed to commit cached model: " + cachedModelPath).toStdString());
    }

    if (!fileMatchesHash(cachedModelPath, expectedSha256)) {
        QFile::remove(cachedModelPath);
        throw std::runtime_error(("Downloaded model hash mismatch: " + cachedModelPath).toStdString());
    }

    return cachedModelPath;
}

bool hasProvider(const std::unordered_set<std::string> &available, const char *provider) {
    return available.find(provider) != available.end();
}

std::string joinProviders(const std::set<std::string> &providers) {
    std::string result;
    for (const std::string &provider : providers) {
        if (!result.empty())
            result += ", ";
        result += provider;
    }
    return result;
}

bool appendDirectMLProvider(Ort::SessionOptions &opts) {
    try {
        Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider(
            opts,
            "DmlExecutionProvider",
            nullptr,
            nullptr,
            0
        ));
        return true;
    } catch (const Ort::Exception &err) {
        qWarning() << "Failed to append DirectML provider by runtime provider name:" << err.what();
    }

    try {
        Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider(
            opts,
            "DML",
            nullptr,
            nullptr,
            0
        ));
        return true;
    } catch (const Ort::Exception &err) {
        qWarning() << "Failed to append DirectML provider:" << err.what();
        return false;
    }
}

Ort::SessionOptions makeSessionOptions(bool allowGpu) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    std::set<std::string> sortedProviders(availableProviders.begin(), availableProviders.end());
    std::unordered_set<std::string> availableSet(availableProviders.begin(), availableProviders.end());

    qDebug() << "Available ONNX Runtime providers:"
             << QString::fromStdString(joinProviders(sortedProviders));

    if (!allowGpu) {
        qDebug() << "Using ONNX Runtime provider: CPUExecutionProvider";
        return opts;
    }

    if (hasProvider(availableSet, "CUDAExecutionProvider")) {
        try {
            OrtCUDAProviderOptions cudaOptions{};
            opts.AppendExecutionProvider_CUDA(cudaOptions);
            qDebug() << "Using ONNX Runtime provider: CUDAExecutionProvider";
            return opts;
        } catch (const Ort::Exception &err) {
            qWarning() << "Failed to append CUDA provider:" << err.what();
        }
    }

    if (hasProvider(availableSet, "DmlExecutionProvider") || hasProvider(availableSet, "DML")) {
        if (appendDirectMLProvider(opts)) {
            qDebug() << "Using ONNX Runtime provider: DmlExecutionProvider";
            return opts;
        }
    }

    qDebug() << "Using ONNX Runtime provider: CPUExecutionProvider";
    return opts;
}

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
    const int channels = mat.channels();
    std::vector<float> values(channels * h * w);

    for (int y = 0; y < h; ++y) {
        const uchar *line = mat.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            const int idx = y * w + x;
            for (int channel = 0; channel < channels; ++channel)
                values[channel * h * w + idx] = line[x * channels + channel] / 255.0f;
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

std::vector<int> genTileStarts(int length, int window, int overlap) {
    const int stride = std::max(1, window - overlap);
    std::vector<int> starts;

    for (int pos = 0; pos < std::max(1, length - window + 1); pos += stride)
        starts.push_back(pos);

    const int finalStart = std::max(0, length - window);
    if (starts.empty() || starts.back() != finalStart)
        starts.push_back(finalStart);

    return starts;
}

std::vector<TileBox> selectChunkTiles(const QImage &baseMaskImage) {
    QImage grayStorage = baseMaskImage.convertToFormat(QImage::Format_Grayscale8);
    cv::Mat maskGray(
        grayStorage.height(),
        grayStorage.width(),
        CV_8UC1,
        grayStorage.bits(),
        static_cast<size_t>(grayStorage.bytesPerLine())
    );

    cv::Mat binMask;
    cv::threshold(maskGray, binMask, ChunkBinThresh, 1, cv::THRESH_BINARY);

    cv::Mat invBinMask = 1 - binMask;
    cv::Mat distFg;
    cv::Mat distBg;
    cv::distanceTransform(binMask, distFg, cv::DIST_L2, 3);
    cv::distanceTransform(invBinMask, distBg, cv::DIST_L2, 3);

    cv::Mat distToBoundary;
    cv::min(distFg, distBg, distToBoundary);

    const int h = maskGray.rows;
    const int w = maskGray.cols;
    const std::vector<int> xs = genTileStarts(w, ChunkTileSize, ChunkOverlap);
    const std::vector<int> ys = genTileStarts(h, ChunkTileSize, ChunkOverlap);

    const int total = ChunkTileSize * ChunkTileSize;
    std::vector<std::pair<TileBox, float>> edgeTiles;
    std::vector<TileBox> posPureTiles;
    std::vector<TileBox> negPureTiles;

    for (int y0 : ys) {
        for (int x0 : xs) {
            const int x1 = x0 + ChunkTileSize;
            const int y1 = y0 + ChunkTileSize;
            if (x1 > w || y1 > h)
                continue;

            cv::Mat tileBin = binMask(cv::Rect(x0, y0, ChunkTileSize, ChunkTileSize));
            const int pos = cv::countNonZero(tileBin);
            const int neg = total - pos;
            const float posFrac = static_cast<float>(pos) / static_cast<float>(total);

            if (posFrac < ChunkMinPosFrac)
                continue;

            TileBox box{x0, y0, x1, y1};
            if (pos == total) {
                posPureTiles.push_back(box);
                continue;
            }

            if (neg == total) {
                negPureTiles.push_back(box);
                continue;
            }

            cv::Mat edgeMask;
            cv::compare(
                distToBoundary(cv::Rect(x0, y0, ChunkTileSize, ChunkTileSize)),
                ChunkEdgeBandPx,
                edgeMask,
                cv::CMP_LE
            );
            const float edgeFrac = static_cast<float>(cv::countNonZero(edgeMask)) / static_cast<float>(total);
            edgeTiles.push_back({box, edgeFrac});
        }
    }

    std::sort(edgeTiles.begin(), edgeTiles.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.second > rhs.second;
    });

    const int poolSize = static_cast<int>(edgeTiles.size() + posPureTiles.size() + negPureTiles.size());
    if (poolSize == 0)
        return {};

    const float weightSum = ChunkEdgeTargetFrac + ChunkPosPureFrac + ChunkNegPureFrac;
    const int targetEdge = std::min(
        static_cast<int>(edgeTiles.size()),
        static_cast<int>(std::lround((ChunkEdgeTargetFrac / weightSum) * poolSize))
    );
    const int targetPos = std::min(
        static_cast<int>(posPureTiles.size()),
        static_cast<int>(std::lround((ChunkPosPureFrac / weightSum) * poolSize))
    );
    const int targetNeg = std::min(
        static_cast<int>(negPureTiles.size()),
        static_cast<int>(std::lround((ChunkNegPureFrac / weightSum) * poolSize))
    );

    std::vector<TileBox> picked;
    for (int i = 0; i < targetEdge; ++i)
        picked.push_back(edgeTiles[i].first);
    picked.insert(picked.end(), posPureTiles.begin(), posPureTiles.begin() + targetPos);
    picked.insert(picked.end(), negPureTiles.begin(), negPureTiles.begin() + targetNeg);

    for (int i = targetEdge; i < static_cast<int>(edgeTiles.size()) && static_cast<int>(picked.size()) < poolSize; ++i)
        picked.push_back(edgeTiles[i].first);
    for (int i = targetPos; i < static_cast<int>(posPureTiles.size()) && static_cast<int>(picked.size()) < poolSize; ++i)
        picked.push_back(posPureTiles[i]);
    for (int i = targetNeg; i < static_cast<int>(negPureTiles.size()) && static_cast<int>(picked.size()) < poolSize; ++i)
        picked.push_back(negPureTiles[i]);

    return picked;
}

cv::Mat makeBlendWindow(int size) {
    cv::Mat win1d(1, size, CV_32FC1);
    for (int i = 0; i < size; ++i) {
        const float value = 0.5f - 0.5f * std::cos(
            2.0f * static_cast<float>(CV_PI) * static_cast<float>(i) / static_cast<float>(size - 1)
        );
        win1d.at<float>(0, i) = std::max(value, 1e-3f);
    }

    cv::Mat win2d = win1d.t() * win1d;
    double maxValue = 1.0;
    cv::minMaxLoc(win2d, nullptr, &maxValue);
    win2d /= static_cast<float>(maxValue);
    return win2d;
}

cv::Mat qImageToGrayMatCopy(const QImage &image) {
    QImage gray = image.convertToFormat(QImage::Format_Grayscale8);
    cv::Mat mat(
        gray.height(),
        gray.width(),
        CV_8UC1,
        gray.bits(),
        static_cast<size_t>(gray.bytesPerLine())
    );
    return mat.clone();
}

class OnnxSegmentationSession : public SegmentationSession {
public:
    OnnxSegmentationSession(const QString &modelPath, Ort::Env &env, int fallbackInputSize, bool allowGpu)
        : inputSize(fallbackInputSize),
          session(env, modelPath.toStdString().c_str(), makeSessionOptions(allowGpu)) {
        readInputSize(fallbackInputSize);
    }

protected:
    void readInputSize(int fallbackInputSize) {
        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> inputShape = inputTensorInfo.GetShape();

        if (inputShape.size() == 4 && inputShape[2] > 0 && inputShape[3] > 0)
            inputSize = static_cast<int>(inputShape[2]);
        else
            inputSize = fallbackInputSize;

        qDebug() << "Model input size:" << inputSize;
    }

    Ort::Value run(const cv::Mat &inputMat, std::vector<float> &inputTensorValues) {
        inputTensorValues = matToNchwFloat(inputMat);

        std::array<int64_t, 4> shape{1, inputMat.channels(), inputMat.rows, inputMat.cols};
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
    U2NetSession(const QString &modelPath, Ort::Env &env, int fallbackInputSize, QString name)
        : OnnxSegmentationSession(modelPath, env, fallbackInputSize, true),
          modelName(std::move(name)) {}

    QString displayName() const override {
        return modelName;
    }

    QImage predictMask(const QImage &img) override {
        QImage rgbStorage;
        cv::Mat originalMat = imageToRgbMat(img, rgbStorage);

        return predictMaskFromMat(originalMat);
    }

    QImage predictMaskFromMat(const cv::Mat &originalMat) {
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
                const float value = range > std::numeric_limits<float>::epsilon()
                    ? (outData[idx] - minValue) / range
                    : outData[idx];
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
    explicit BiRefNetSession(const QString &modelPath, Ort::Env &env, bool allowGpu)
        : OnnxSegmentationSession(modelPath, env, 1440, allowGpu) {}

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

class U2NetPChunkRefinementSession final : public ChunkRefinementSession {
public:
    explicit U2NetPChunkRefinementSession(const QString &modelPath, Ort::Env &env)
        : tileSession(modelPath, env, ChunkTileSize, "U2NetP Chunks") {}

    QImage refineMask(
        const QImage &img,
        const QImage &baseMask,
        const std::function<void(int, int)> &progressCallback
    ) override {
        std::vector<TileBox> boxes = selectChunkTiles(baseMask);
        if (boxes.empty()) {
            qWarning() << "No refinement tiles selected; returning empty mask";
            cv::Mat emptyMask = cv::Mat::zeros(baseMask.height(), baseMask.width(), CV_8UC1);
            return grayscaleMatToImageCopy(emptyMask);
        }

        QImage rgbStorage;
        cv::Mat originalMat = imageToRgbMat(img, rgbStorage);
        if (originalMat.cols < ChunkTileSize || originalMat.rows < ChunkTileSize) {
            qWarning() << "Image smaller than chunk size; returning empty mask";
            cv::Mat emptyMask = cv::Mat::zeros(originalMat.rows, originalMat.cols, CV_8UC1);
            return grayscaleMatToImageCopy(emptyMask);
        }

        cv::Mat acc = cv::Mat::zeros(originalMat.rows, originalMat.cols, CV_32FC1);
        cv::Mat weightSum = cv::Mat::zeros(originalMat.rows, originalMat.cols, CV_32FC1);
        cv::Mat blendWindow = makeBlendWindow(ChunkTileSize);
        cv::Mat baseMaskMat = qImageToGrayMatCopy(baseMask);

        qDebug() << "Running chunk refinement on" << boxes.size() << "tiles";
        for (int tileIndex = 0; tileIndex < static_cast<int>(boxes.size()); ++tileIndex) {
            if (progressCallback)
                progressCallback(tileIndex + 1, static_cast<int>(boxes.size()));

            const TileBox &box = boxes[tileIndex];
            cv::Mat tileMat = originalMat(cv::Rect(box.x0, box.y0, ChunkTileSize, ChunkTileSize)).clone();
            cv::Mat tileBaseMask = baseMaskMat(cv::Rect(box.x0, box.y0, ChunkTileSize, ChunkTileSize)).clone();

            std::vector<cv::Mat> rgbChannels;
            cv::split(tileMat, rgbChannels);
            rgbChannels.push_back(tileBaseMask);

            cv::Mat tileWithMask;
            cv::merge(rgbChannels, tileWithMask);

            QImage tileMaskImage = tileSession.predictMaskFromMat(tileWithMask);
            cv::Mat tileMask = qImageToGrayMatCopy(tileMaskImage);
            if (tileMask.cols != ChunkTileSize || tileMask.rows != ChunkTileSize) {
                cv::resize(
                    tileMask,
                    tileMask,
                    cv::Size(ChunkTileSize, ChunkTileSize),
                    0.0,
                    0.0,
                    cv::INTER_LANCZOS4
                );
            }

            cv::Mat tileFloat;
            tileMask.convertTo(tileFloat, CV_32FC1);

            cv::Rect roi(box.x0, box.y0, ChunkTileSize, ChunkTileSize);
            acc(roi) += tileFloat.mul(blendWindow);
            weightSum(roi) += blendWindow;
        }

        cv::Mat refinedFloat;
        cv::divide(acc, cv::max(weightSum, 1e-6f), refinedFloat);

        cv::Mat refinedMask;
        refinedFloat.convertTo(refinedMask, CV_8UC1);

        cv::morphologyEx(
            refinedMask,
            refinedMask,
            cv::MORPH_OPEN,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3))
        );
        cv::GaussianBlur(refinedMask, refinedMask, cv::Size(3, 3), 1.0, 1.0, cv::BORDER_DEFAULT);

        return grayscaleMatToImageCopy(refinedMask);
    }

private:
    U2NetSession tileSession;
};

} // namespace

std::unique_ptr<SegmentationSession> createSegmentationSession(
    ModelKind kind,
    Ort::Env &env,
    bool allowGpu,
    const std::function<void(const QString &)> &statusCallback
) {
    switch (kind) {
    case ModelKind::U2Net:
        return std::make_unique<U2NetSession>(
            ensureModelAvailable(
                "u2net.onnx",
                "https://github.com/83catsonthemoon/medusab-segmenter/releases/download/v0.0.0/u2net.onnx",
                "ac44f925c222a842d51d60f336e766621fc3593bced20a3624663dc0022c97ed",
                statusCallback
            ),
            env,
            1024,
            "U2Net"
        );
    case ModelKind::BiRefNet:
        return std::make_unique<BiRefNetSession>(
            ensureModelAvailable(
                "birefnet.onnx",
                "https://github.com/83catsonthemoon/medusab-segmenter/releases/download/v0.0.0/birefnet.onnx",
                "9f765b0372222b0ea2dcd60a013e8a2e77a905f1ff2f65c3ccdc24c3a1202251",
                statusCallback
            ),
            env,
            allowGpu
        );
    case ModelKind::U2NetPChunks:
        return std::make_unique<U2NetSession>(
            ensureModelAvailable(
                "u2netp_chunks.onnx",
                "https://github.com/83catsonthemoon/medusab-segmenter/releases/download/v0.0.0/u2netp_chunks.onnx",
                "657bdf94e7f1a66d8f0d36b645023cb06368c48b48b47f7c6502d835d5457b49",
                statusCallback
            ),
            env,
            512,
            "U2NetP Chunks"
        );
    }

    throw std::invalid_argument("Unknown model kind");
}

std::unique_ptr<ChunkRefinementSession> createChunkRefinementSession(
    Ort::Env &env,
    const std::function<void(const QString &)> &statusCallback
) {
    return std::make_unique<U2NetPChunkRefinementSession>(
        ensureModelAvailable(
            "u2netp_chunks.onnx",
            "https://github.com/83catsonthemoon/medusab-segmenter/releases/download/v0.0.0/u2netp_chunks.onnx",
            "657bdf94e7f1a66d8f0d36b645023cb06368c48b48b47f7c6502d835d5457b49",
            statusCallback
        ),
        env
    );
}
