#include <QApplication>
#include <QImageReader>
#include "mainwindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    QImageReader::setAllocationLimit(0);
    MainWindow w;
    w.show();
    return app.exec();
}
