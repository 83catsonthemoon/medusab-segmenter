#include "mainwindow.h"

#include <QDebug>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QHBoxLayout>
#include <QMimeData>
#include <QApplication>
#include <QResizeEvent>
#include <QSizePolicy>
#include <QStackedLayout>
#include <QVBoxLayout>

MainWindow::MainWindow()
    : modelComboBox(nullptr),
      inputLabel(nullptr),
      outputLabel(nullptr),
      outputStatusLabel(nullptr),
      env(ORT_LOGGING_LEVEL_WARNING, "medusab-segmenter"),
      segmentationSession(nullptr),
      currentModelKind(ModelKind::U2Net) {

    setWindowTitle("Medusab Segmenter");
    setAcceptDrops(true);

    auto *central = new QWidget();
    auto *mainLayout = new QVBoxLayout(central);
    auto *imageLayout = new QHBoxLayout();

    modelComboBox = new QComboBox();
    modelComboBox->addItem("U2Net", static_cast<int>(ModelKind::U2Net));
    modelComboBox->addItem("BiRefNet", static_cast<int>(ModelKind::BiRefNet));

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

    mainLayout->addWidget(modelComboBox);
    mainLayout->addLayout(imageLayout, 1);

    setCentralWidget(central);

    connect(modelComboBox, &QComboBox::currentIndexChanged, this, [this]() {
        const ModelKind nextModelKind = selectedModelKind();
        if (nextModelKind == currentModelKind)
            return;

        currentModelKind = nextModelKind;
        segmentationSession.reset();
        currentOutputImage = {};

        if (!currentInputImage.isNull())
            runCurrentModel();
        else
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

    QImage img(path);
    if (img.isNull()) {
        qWarning() << "Failed to load image:" << path;
        return;
    }

    currentInputImage = img;
    currentOutputImage = {};
    updateImageLabels();
    runCurrentModel();
}

void MainWindow::resizeEvent(QResizeEvent *event) {
    QMainWindow::resizeEvent(event);
    updateImageLabels();
}

ModelKind MainWindow::selectedModelKind() const {
    return static_cast<ModelKind>(modelComboBox->currentData().toInt());
}

void MainWindow::ensureSession() {
    if (segmentationSession)
        return;

    setOutputStatus("Loading model...");
    segmentationSession = createSegmentationSession(currentModelKind, env);
}

void MainWindow::runCurrentModel() {
    if (currentInputImage.isNull())
        return;

    try {
        ensureSession();
        setOutputStatus("Running model...");
        currentOutputImage = segmentationSession->predictMask(currentInputImage);
    } catch (const Ort::Exception &err) {
        qWarning() << "ONNX Runtime inference failed:" << err.what();
        currentOutputImage = {};
        setOutputStatus("Inference failed");
        return;
    } catch (const std::exception &err) {
        qWarning() << "Inference failed:" << err.what();
        currentOutputImage = {};
        setOutputStatus("Inference failed");
        return;
    }

    setOutputStatus({});
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
