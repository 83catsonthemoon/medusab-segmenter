#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QImage>
#include <QComboBox>
#include <onnxruntime_cxx_api.h>

#include <memory>

#include "segmentationsession.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow();

protected:
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    QComboBox *modelComboBox;
    QLabel *inputLabel;
    QLabel *outputLabel;
    QLabel *outputStatusLabel;

    Ort::Env env;
    std::unique_ptr<SegmentationSession> segmentationSession;
    ModelKind currentModelKind;
    QImage currentInputImage;
    QImage currentOutputImage;

    ModelKind selectedModelKind() const;
    void ensureSession();
    void runCurrentModel();
    void updateImageLabels();
    void setOutputStatus(const QString &status);
};
