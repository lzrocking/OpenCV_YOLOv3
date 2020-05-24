#include "faceDetector.h"
#include <fstream>
#include <QMessageBox>

void FaceDetector::loadYOLO_Cfg(cv::String weightsFilename,
                                cv::String cfgFinlename,
                                cv::String namesFilename)
{
    net=cv::dnn::readNetFromDarknet(cfgFinlename,weightsFilename);
    std::ifstream ifs(namesFilename.c_str());
    if(!ifs.is_open())
    {
        QMessageBox Msg;
        Msg.setText("open YOLO names file failed");
        return;
    }
    std::string line;
    while(std::getline(ifs,line))
    {
        classes.push_back(line);
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

void FaceDetector::loadFiles(cv::String faceCascadeFilename,
                       cv::String eyeCascadeFilename)
{
    // TODO: Add in a try catch statement here
    if( !faceCascade.load( faceCascadeFilename ) )
    {
        std::cout << "Error Loading" << faceCascadeFilename << std::endl;
    }

    if( !eyeCascade.load( eyeCascadeFilename ) )
    {
        std::cout << "Error Loading" << eyeCascadeFilename << std::endl;
    }
}
FaceDetector::~FaceDetector()
{

}

void FaceDetector::SetYOLO(bool mark)
{
    YOLO_Mark=mark;
}

void FaceDetector::processFrame(const cv::Mat &frame)
{
    if(YOLO_Mark)
    {
        //qDebug()<<"using YOLO";
        YOLO_process(frame);
        return;
    }
    if (processAll_)
        process(frame);
    else
        queue(frame);
}

void FaceDetector::setProcessAll(bool all)
{
    processAll_ = all;
}

void FaceDetector::YOLO_process(cv::Mat frame)
{
    // Create a 4D blob from a frame.
    static cv::Mat blob;
    cv::dnn::blobFromImage(frame,
                           blob,
                           double(1.0/255.0),
                           cv::Size(inpWidth, inpHeight),
                           cv::Scalar(0,0,0),
                           true, false);

    net.setInput(blob);
    // Runs the forward pass to get output of the output layers
    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames(net));
    // Remove the bounding boxes with low confidence
    postprocess(frame, outs);
    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    QString tmpString=QString("Inference time for a frame : %1 ms").arg(t,4,'f',2);
    std::string label = tmpString.toStdString();
    putText(frame, label, Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
    // Write the frame with the detection boxes
    static Mat detectedFrame;
    cv::cvtColor(frame, detectedFrame, cv::COLOR_BGR2RGB);
    const QImage image((const unsigned char*)detectedFrame.data,
                       frame.cols, frame.rows, detectedFrame.step,
                       QImage::Format_RGB888, &matDeleter,
                       new cv::Mat(detectedFrame));
    image.rgbSwapped();
    Q_ASSERT(image.constBits() == detectedFrame.data);
    emit image_signal(image);
}

void FaceDetector::process(cv::Mat frame)
{
    cv::Mat grey_image;
    cv::cvtColor(frame, grey_image, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(grey_image, grey_image);

    std::vector<cv::Rect> faces;
    // Calculate the camera size and set the size to 1/8 of screen height
    faceCascade.detectMultiScale(grey_image, faces, 1.1, 2,  0|cv::CASCADE_SCALE_IMAGE ,
                                 cv::Size(frame.cols/4, frame.rows/4)); // Minimum size of obj
    //-- Draw rectangles around faces
    for( size_t i = 0; i < faces.size(); i++)
    {
        cv::rectangle(frame, faces[i], cv::Scalar( 255, 0, 255 ));
        /*
        cv::Point center( faces[i].x + faces[i].width*0.5,
                  faces[i].y + faces[i].height*0.5);

        ellipse( frame, center,
             cv::Size( faces[i].width*0.5, faces[i].height*0.5 ),
             0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0);

        cv::Mat faceROI = frameGray( faces[i] );
        std::vector<cv::Rect> eyes;

        //-- In each face, detect eyes
        eyeCascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

        for( size_t j = 0; j < eyes.size(); j++)
        {
            cv::Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5,
                      faces[i].y + eyes[j].y + eyes[j].height*0.5 );
            int radius = cvRound( (eyes[j].width + eyes[j].height) *0.25);
            circle( frame, center, radius, cv::Scalar( 255, 0, 0 ), 4, 8, 0);
        }
        */

    }
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    const QImage image((const unsigned char*)frame.data, frame.cols, frame.rows, frame.step,
                       QImage::Format_RGB888, &matDeleter, new cv::Mat(frame));
    image.rgbSwapped();
    Q_ASSERT(image.constBits() == frame.data);
    emit image_signal(image);
}
void FaceDetector::timerEvent(QTimerEvent *ev)
{
    if (ev->timerId() != timer_.timerId())
        return;
    process(frame_);
    frame_.release();
    timer_.stop();
}

void FaceDetector::queue(const cv::Mat &frame)
{
    if (!frame.empty())
        qDebug() << "Converter dropped frame !";

    frame_ = frame;
    if (!timer_.isActive())
        timer_.start(0, this);
}


void FaceDetector::matDeleter(void *mat)
{
    delete static_cast<cv::Mat*>(mat);
}

void FaceDetector::facecascade_filename(QString filename)
{
    cv::String temp = filename.toStdString().c_str();
    if( !faceCascade.load( temp ) )
    {
        std::cout << "Error Loading" << filename.toStdString() << std::endl;
    }
    facecascade_filename_ = filename;
    // FIXME: Incorrect Implementation
    loadFiles(filename.toStdString().c_str(), filename.toStdString().c_str());
}

vector<String> FaceDetector::getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void FaceDetector::postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

// Draw the predicted bounding box
void FaceDetector::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame,
                  Point(left, top),
                  Point(right, bottom),
                  Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    QString tmpString=QString("%1").arg(conf,4,'f',2);
    std::string label = tmpString.toStdString();
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), cv::FILLED);
    putText(frame, label, Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}
