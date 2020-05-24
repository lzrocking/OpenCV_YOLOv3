#include <QObject>
#include <QBasicTimer>
#include <QTimerEvent>
#include <QDir>
#include <QDebug>
#include <QImage>
#include <QString>
#include <QResource>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

using cv::dnn::Net;
using cv::String;
using cv::Scalar;
using cv::Point;
using cv::Rect;
using cv::Size;
using cv::Mat;
using std::vector;


class FaceDetector : public QObject
{
    Q_OBJECT
    QString facecascade_filename_;
    QString eyecascade_filename_;
    QString YOLO_weights_filename;
    QString YOLO_cfg_filename;
    QString YOLO_names_filename;
    std::vector<std::string> classes;
    QBasicTimer timer_;
    cv::Mat frame_;
    bool processAll_;
    bool YOLO_Mark;
    float confThreshold;
    float nmsThreshold;
    int inpWidth;
    int inpHeight;
    Net net;
    cv::CascadeClassifier faceCascade;
    cv::CascadeClassifier eyeCascade;
    void process(cv::Mat frame);
    void YOLO_process(cv::Mat frame);
    void loadFiles(cv::String faceCascadeFilename, cv::String eyesCascadeFilename);
    void loadYOLO_Cfg(cv::String weightsFilename,
                      cv::String cfgFinlename,
                      cv::String namesFilename);
    std::vector<cv::String> getOutputsNames(const Net& net);
    void postprocess(Mat& frame, const vector<Mat>& outs);
    void drawPred(int classId, float conf,
                  int left, int top,
                  int right, int bottom,
                  Mat& frame);
    void queue(const cv::Mat & frame);
    void timerEvent(QTimerEvent* ev);
    static void matDeleter(void* mat);

public:
    FaceDetector(QObject *parent=0) : QObject(parent), processAll_(true)
    {
        facecascade_filename_ = "resources/haarcascade_frontalface_default.xml";
        eyecascade_filename_ = "resources/haarcascade_eye.xml";
        loadFiles(facecascade_filename_.toStdString().c_str(),
                  eyecascade_filename_.toStdString().c_str());
        YOLO_weights_filename="yolo-coco/yolov3.weights";
        YOLO_cfg_filename="yolo-coco/yolov3.cfg";
        YOLO_names_filename="yolo-coco/coco.names";
        loadYOLO_Cfg(YOLO_weights_filename.toStdString().c_str(),
                     YOLO_cfg_filename.toStdString().c_str(),
                     YOLO_names_filename.toStdString().c_str());
        confThreshold=0.5F;
        nmsThreshold=0.4F;
        inpWidth=416;
        inpHeight=416;
        YOLO_Mark=false;
    }
    void setProcessAll(bool all);
    ~FaceDetector();

signals:
    void image_signal(const QImage&);

public slots:
    void processFrame(const cv::Mat& frame);
    void facecascade_filename(QString filename);
    void SetYOLO(bool mark);
};
