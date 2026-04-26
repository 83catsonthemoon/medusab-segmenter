#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QImage>
#include <QComboBox>
#include <QCheckBox>
#include <QPushButton>
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
    QCheckBox *hurtComputerCheckBox;
    QCheckBox *nasaCheckBox;
    QPushButton *loadImageButton;
    QPushButton *runButton;
    QPushButton *saveMaskButton;
    QLabel *inputLabel;
    QLabel *outputLabel;
    QLabel *outputStatusLabel;

    Ort::Env env;
    std::unique_ptr<SegmentationSession> segmentationSession;
    std::unique_ptr<ChunkRefinementSession> chunkRefinementSession;
    ModelKind currentModelKind;
    QImage currentInputImage;
    QImage currentOutputImage;

    ModelKind selectedModelKind() const;
    bool confirmHeavyModel(ModelKind modelKind);
    bool useGpuForSelectedModel() const;
    void loadImageFromPath(const QString &path);
    void openImageDialog();
    void saveMaskDialog();
    void ensureSession();
    void ensureChunkRefinementSession();
    void runCurrentModel();
    void updateImageLabels();
    void updateButtonStates();
    void setOutputStatus(const QString &status);
};
