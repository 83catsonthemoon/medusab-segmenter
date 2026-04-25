#pragma once

#include <QImage>
#include <QString>

#include <onnxruntime_cxx_api.h>

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

std::unique_ptr<SegmentationSession> createSegmentationSession(
    ModelKind kind,
    Ort::Env &env
);

