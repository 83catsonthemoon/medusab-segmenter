#include "mainwindow.h"

#include <QDebug>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QMimeData>
#include <QApplication>
#include <QMessageBox>
#include <QResizeEvent>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QStackedLayout>
#include <QVBoxLayout>

MainWindow::MainWindow()
    : modelComboBox(nullptr),
      hurtComputerCheckBox(nullptr),
      nasaCheckBox(nullptr),
      loadImageButton(nullptr),
      runButton(nullptr),
      saveMaskButton(nullptr),
      inputLabel(nullptr),
      outputLabel(nullptr),
      outputStatusLabel(nullptr),
      env(ORT_LOGGING_LEVEL_WARNING, "medusab-segmenter"),
      segmentationSession(nullptr),
      chunkRefinementSession(nullptr),
      currentModelKind(ModelKind::U2Net) {

    setWindowTitle("Medusab Segmenter");
    setAcceptDrops(true);

    auto *central = new QWidget();
    auto *mainLayout = new QVBoxLayout(central);
    auto *controlsLayout = new QHBoxLayout();
    auto *imageLayout = new QHBoxLayout();

    modelComboBox = new QComboBox();
    modelComboBox->addItem("U2Net", static_cast<int>(ModelKind::U2Net));
    modelComboBox->addItem("BiRefNet", static_cast<int>(ModelKind::BiRefNet));

    hurtComputerCheckBox = new QCheckBox("Make my computer hurt");
    hurtComputerCheckBox->setToolTip(
        "Runs expensive chunked U2NetP refinement after the base model."
    );
    nasaCheckBox = new QCheckBox("I work at NASA");
    nasaCheckBox->setToolTip("Runs BiRefNet on GPU, which requires a lot of VRAM");
    nasaCheckBox->hide();

    loadImageButton = new QPushButton("Load image");
    runButton = new QPushButton("Run");
    runButton->setEnabled(false);
    saveMaskButton = new QPushButton("Save mask");
    saveMaskButton->setEnabled(false);

    inputLabel = new QLabel("Drop Image");
    inputLabel->setMinimumSize(240, 240);
    inputLabel->setStyleSheet("border: 1px solid gray;");
    inputLabel->setAlignment(Qt::AlignCenter);
    inputLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    outputLabel = new QLabel("Output");
    outputLabel->setMinimumSize(240, 240);
    outputLabel->setStyleSheet("border: 1px solid gray;");
    outputLabel->setAlignment(Qt::AlignCenter);
    outputLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    outputStatusLabel = new QLabel();
    outputStatusLabel->setAlignment(Qt::AlignCenter);
    outputStatusLabel->setStyleSheet(
        "background-color: rgba(0, 0, 0, 170);"
        "color: white;"
        "font-weight: 600;"
        "padding: 12px;"
    );
    outputStatusLabel->hide();

    auto *outputPane = new QWidget();
    outputPane->setMinimumSize(240, 240);
    outputPane->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    auto *outputStack = new QStackedLayout(outputPane);
    outputStack->setStackingMode(QStackedLayout::StackAll);
    outputStack->setContentsMargins(0, 0, 0, 0);
    outputStack->addWidget(outputLabel);
    outputStack->addWidget(outputStatusLabel);

    imageLayout->addWidget(inputLabel, 1);
    imageLayout->addWidget(outputPane, 1);

    controlsLayout->addWidget(modelComboBox);
    controlsLayout->addWidget(hurtComputerCheckBox);
    controlsLayout->addWidget(nasaCheckBox);
    controlsLayout->addWidget(loadImageButton);
    controlsLayout->addWidget(runButton);
    controlsLayout->addWidget(saveMaskButton);
    controlsLayout->addStretch(1);

    mainLayout->addLayout(controlsLayout);
    mainLayout->addLayout(imageLayout, 1);

    setCentralWidget(central);

    connect(modelComboBox, &QComboBox::currentIndexChanged, this, [this]() {
        const ModelKind nextModelKind = selectedModelKind();
        if (nextModelKind == currentModelKind)
            return;

        if (!confirmHeavyModel(nextModelKind)) {
            QSignalBlocker blocker(modelComboBox);
            modelComboBox->setCurrentIndex(modelComboBox->findData(static_cast<int>(currentModelKind)));
            return;
        }

        currentModelKind = nextModelKind;
        segmentationSession.reset();
        currentOutputImage = {};
        nasaCheckBox->setVisible(currentModelKind == ModelKind::BiRefNet);
        updateButtonStates();
        updateImageLabels();
    });

    connect(loadImageButton, &QPushButton::clicked, this, &MainWindow::openImageDialog);
    connect(runButton, &QPushButton::clicked, this, &MainWindow::runCurrentModel);
    connect(saveMaskButton, &QPushButton::clicked, this, &MainWindow::saveMaskDialog);
    connect(hurtComputerCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        Q_UNUSED(checked);
        currentOutputImage = {};
        updateButtonStates();
        updateImageLabels();
    });
    connect(nasaCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        Q_UNUSED(checked);
        segmentationSession.reset();
        currentOutputImage = {};
        updateButtonStates();
        updateImageLabels();
    });
}

void MainWindow::dragEnterEvent(QDragEnterEvent *event) {
    if (event->mimeData()->hasUrls())
        event->acceptProposedAction();
}

void MainWindow::dropEvent(QDropEvent *event) {
    auto urls = event->mimeData()->urls();
    if (urls.isEmpty())
        return;

    QString path = urls.first().toLocalFile();
    if (path.isEmpty())
        return;

    loadImageFromPath(path);
}

void MainWindow::resizeEvent(QResizeEvent *event) {
    QMainWindow::resizeEvent(event);
    updateImageLabels();
}

ModelKind MainWindow::selectedModelKind() const {
    return static_cast<ModelKind>(modelComboBox->currentData().toInt());
}

bool MainWindow::confirmHeavyModel(ModelKind modelKind) {
    if (modelKind != ModelKind::BiRefNet)
        return true;

    return QMessageBox::warning(
        this,
        "BiRefNet resource warning",
        "BiRefNet is more accurate, but it can take roughly 24 GB of memory to run. "
        "U2Net is the sane, moderate-resource model.\n\nUse BiRefNet anyway?",
        QMessageBox::Ok | QMessageBox::Cancel,
        QMessageBox::Cancel
    ) == QMessageBox::Ok;
}

bool MainWindow::useGpuForSelectedModel() const {
    if (currentModelKind == ModelKind::BiRefNet)
        return nasaCheckBox->isChecked();

    return true;
}

void MainWindow::loadImageFromPath(const QString &path) {
    QImage img(path);
    if (img.isNull()) {
        qWarning() << "Failed to load image:" << path;
        return;
    }

    currentInputImage = img;
    currentOutputImage = {};
    updateButtonStates();
    updateImageLabels();
}

void MainWindow::openImageDialog() {
    const QString path = QFileDialog::getOpenFileName(
        this,
        "Load image",
        QString(),
        "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All files (*)"
    );

    if (!path.isEmpty())
        loadImageFromPath(path);
}

void MainWindow::saveMaskDialog() {
    if (currentOutputImage.isNull())
        return;

    QString path = QFileDialog::getSaveFileName(
        this,
        "Save mask",
        "mask.png",
        "PNG image (*.png);;All files (*)"
    );

    if (path.isEmpty())
        return;

    if (!path.endsWith(".png", Qt::CaseInsensitive))
        path += ".png";

    QImage mask = currentOutputImage.convertToFormat(QImage::Format_Grayscale8);
    if (!mask.save(path, "PNG"))
        qWarning() << "Failed to save mask:" << path;
}

void MainWindow::ensureSession() {
    if (segmentationSession)
        return;

    setOutputStatus("Loading model...");
    segmentationSession = createSegmentationSession(
        currentModelKind,
        env,
        useGpuForSelectedModel(),
        [this](const QString &status) { setOutputStatus(status); }
    );
}

void MainWindow::ensureChunkRefinementSession() {
    if (chunkRefinementSession)
        return;

    setOutputStatus("Loading refinement model...");
    chunkRefinementSession = createChunkRefinementSession(
        env,
        [this](const QString &status) { setOutputStatus(status); }
    );
}

void MainWindow::runCurrentModel() {
    if (currentInputImage.isNull())
        return;

    try {
        ensureSession();
        setOutputStatus("Running model...");
        currentOutputImage = segmentationSession->predictMask(currentInputImage);

        if (hurtComputerCheckBox->isChecked()) {
            updateButtonStates();
            updateImageLabels();
            ensureChunkRefinementSession();
            setOutputStatus("Refining mask...");
            currentOutputImage = chunkRefinementSession->refineMask(
                currentInputImage,
                currentOutputImage,
                [this](int tileIndex, int tileCount) {
                    setOutputStatus(QString("Refining mask... Tile %1/%2").arg(tileIndex).arg(tileCount));
                }
            );
        }
    } catch (const Ort::Exception &err) {
        qWarning() << "ONNX Runtime inference failed:" << err.what();
        currentOutputImage = {};
        segmentationSession.reset();
        chunkRefinementSession.reset();
        updateButtonStates();
        setOutputStatus("Inference failed");
        return;
    } catch (const std::exception &err) {
        qWarning() << "Inference failed:" << err.what();
        currentOutputImage = {};
        segmentationSession.reset();
        chunkRefinementSession.reset();
        updateButtonStates();
        setOutputStatus("Inference failed");
        return;
    }

    segmentationSession.reset();
    chunkRefinementSession.reset();
    setOutputStatus({});
    updateButtonStates();
    updateImageLabels();
}

void MainWindow::updateImageLabels() {
    if (!currentInputImage.isNull()) {
        inputLabel->setPixmap(QPixmap::fromImage(currentInputImage).scaled(
            inputLabel->contentsRect().size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation
        ));
    } else {
        inputLabel->setText("Drop Image");
    }

    if (!currentOutputImage.isNull()) {
        outputLabel->setPixmap(QPixmap::fromImage(currentOutputImage).scaled(
            outputLabel->contentsRect().size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation
        ));
    } else {
        outputLabel->setText("Output");
        outputLabel->setPixmap({});
    }
}

void MainWindow::updateButtonStates() {
    runButton->setEnabled(!currentInputImage.isNull());
    saveMaskButton->setEnabled(!currentOutputImage.isNull());
}

void MainWindow::setOutputStatus(const QString &status) {
    if (status.isEmpty()) {
        outputStatusLabel->hide();
    } else {
        outputStatusLabel->setText(status);
        outputStatusLabel->show();
        outputStatusLabel->raise();
    }

    QApplication::processEvents();
}
