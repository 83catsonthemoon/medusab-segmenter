#pragma once

#include <QImage>
#include <QString>

#include <onnxruntime_cxx_api.h>

#include <functional>
#include <memory>

enum class ModelKind {
    U2Net,
    BiRefNet,
    U2NetPChunks,
};

class SegmentationSession {
public:
    virtual ~SegmentationSession() = default;

    virtual QString displayName() const = 0;
    virtual QImage predictMask(const QImage &img) = 0;
};

class ChunkRefinementSession {
public:
    virtual ~ChunkRefinementSession() = default;

    virtual QImage refineMask(
        const QImage &img,
        const QImage &baseMask,
        const std::function<void(int, int)> &progressCallback
    ) = 0;
};

std::unique_ptr<SegmentationSession> createSegmentationSession(
    ModelKind kind,
    Ort::Env &env,
    bool allowGpu = true,
    const std::function<void(const QString &)> &statusCallback = {}
);

std::unique_ptr<ChunkRefinementSession> createChunkRefinementSession(
    Ort::Env &env,
    const std::function<void(const QString &)> &statusCallback = {}
);
