#-------------------------------------------------
#
# Project created by QtCreator 2015-03-25T18:15:50
#
#-------------------------------------------------

QT += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
CONFIG += c++11
TARGET = opencv
TEMPLATE = app
DESTDIR = $$PWD
INCLUDEPATH +=  ./resources \
                D:/opencv-4.2.0/build_CUDA/install/include \
#                D:/OpenCV/build/install/include


Debug:{LIBS += -LD:/opencv-4.2.0/build_CUDA/install/x64/vc15/lib \
                -LD:/OpenCV/build/install/x64/vc15/lib \
                -lopencv_core420d \
                -lopencv_dnn420d \
                -lopencv_videoio420d \
                -lopencv_objdetect420d \
                -lopencv_imgproc420d \
                }

Release:{LIBS += -LD:/opencv-4.2.0/build_CUDA/install/x64/vc15/lib \
                -lopencv_core420 \
                -lopencv_dnn420 \
                -lopencv_videoio420 \
                -lopencv_objdetect420 \
                -lopencv_imgproc420}

#Release:{LIBS += -LD:/OpenCV/build/install/x64/vc15/lib \
#                -lopencv_core410 \
#                -lopencv_dnn410 \
#                -lopencv_videoio410 \
#                -lopencv_objdetect410 \
#                -lopencv_imgproc410}

SOURCES += main.cpp \
    gui/mainwindow.cpp \
    camera.cpp \
    gui/displaywidget.cpp \
    faceDetector.cpp \
    gui/imageviewer.cpp

HEADERS += gui/mainwindow.h \
    camera.h \
    gui/displaywidget.h \
    faceDetector.h \
    gui/imageviewer.h
