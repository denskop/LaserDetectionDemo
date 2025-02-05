#include <cassert>
#include <iostream>
#include <vector>
#include <opencv/cv.h>
// cv::imread
#include <opencv2/imgcodecs.hpp>
// cv::ColorConversionCodes
// cv::GaussianBlur
#include <opencv2/imgproc.hpp>
// cv::imshow
#include <opencv2/highgui.hpp>
// CV <-> Eigen
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

// Выравнивание гистограммы
cv::Mat applyHistEqualization(const cv::Mat &img, double clipLimit, const cv::Size &gridSize)
{
    cv::Mat imgLab, labComponents[3], imgRes;

    cv::cvtColor(img, imgLab, cv::ColorConversionCodes::COLOR_BGR2Lab);
    cv::split(imgLab, labComponents);

    const cv::Mat &lightness = labComponents[0];
    auto ptr = cv::createCLAHE(clipLimit, gridSize);
    ptr->apply(lightness, lightness);

    cv::merge(labComponents, 3, imgLab);
    cv::cvtColor(imgLab, imgRes, cv::ColorConversionCodes::COLOR_Lab2BGR);

    return imgRes;
}


// Квадрат пикселей
cv::Mat applyPixSqrWithNormCh(const cv::Mat &channel)
{
    assert(channel.type() == CV_8U);

    double delta, min, max;
    minMaxLoc(channel, &min, &max);
    delta = max - min > 0 ? max - min : 1;

    cv::Mat res = channel.clone();
    for (auto p = res.begin<uchar>(); p < res.end<uchar>(); p++)
    {
        *p = uchar(255 * ((pow(int(*p), 2) - min) / delta));
    }

    return res;
}

cv::Mat applyPixSqrWithNorm(const cv::Mat &img)
{
    cv::Mat channels[3], resChannels[3], res;
    cv::split(img, channels);

    resChannels[0] = applyPixSqrWithNormCh(channels[0]);
    for (int i = 1; i < 3; i++)
    {
        resChannels[i] = resChannels[0];
    }

    cv::merge(resChannels, 3, res);
    return res;
}

cv::Mat histEqualizationProcessor(const cv::Mat &img)
{
    return applyHistEqualization(img, 3.0, cv::Size(8, 8));
}

cv::Mat blurProcessor(const cv::Mat &img)
{
    cv::Mat res;
    cv::GaussianBlur(img, res, cv::Size(3, 3), 0);
    return res;
}

cv::Mat pixSqrAndNormProcessor(const cv::Mat &img)
{
    return img;
    // return applyPixSqrWithNorm(img);
}

cv::Mat thresholdFilterProcessor(const cv::Mat &img)
{
    cv::Mat mask, res;
    cv::inRange(img, cv::Scalar(150, 20, 20), cv::Scalar(255, 255, 255), mask);
    cv::bitwise_and(img, img, res, mask);
    return res;
}

cv::Mat erodeDilateProcessor(const cv::Mat &img)
{
    return img;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 2));
    cv::Mat res;
    cv::erode(img, res, kernel);
    cv::dilate(res, res, kernel);
    return res;
}

cv::Mat stubProcessor(const cv::Mat &img)
{
    return img;
}

typedef cv::Mat (*Processor)(const cv::Mat &);

// Подготовка изображения
void prepareImage(const cv::Mat &img, std::vector<cv::Mat> &processed)
{
    std::vector<Processor> stages = {
        histEqualizationProcessor,
        blurProcessor,
        thresholdFilterProcessor,
        erodeDilateProcessor,
        pixSqrAndNormProcessor,
        stubProcessor,
    };

    cv::Mat imgProcessed = img;
    processed.clear();
    for (auto &processor : stages)
    {
        imgProcessed = processor(imgProcessed);
        processed.push_back(imgProcessed);
    }
}

// Нормализация значений
void calcNorm(const Eigen::VectorXi &in, Eigen::VectorXf &out)
{
    int min = in.minCoeff();
    int max = in.maxCoeff();
    int delta = max - min > 0 ? max - min : 1;

    out.resize(in.size());
    int i = 0;
    for (int value : in)
    {
        out[i++] = (float)(value - min) / delta;
    }
}

// Подсчет центра масс
float calcMassCenter(const Eigen::VectorXi &in)
{
    Eigen::VectorXf norm;
    calcNorm(in, norm);
    auto a = Eigen::VectorXf::NullaryExpr([&norm](Eigen::Index i)
                                          { return i * pow(norm(i), 2); });
    auto b = in.unaryExpr([](float x)
                          { return pow(x, 2); });
    return a.sum() / b.sum();
}

// I = f(x)
int f(const cv::Mat &img, int x, float intense_delta_thr = 0.5)
{
    cv::Mat vert_line = img.col(x);
    cv::Mat channels[3];
    cv::split(vert_line, channels);

    Eigen::VectorXf red_channel;
    cv::cv2eigen(channels[0], red_channel);

    Eigen::Index maxIndex;

    // TODO: Считать по формуле Гаусса
    red_channel.maxCoeff(&maxIndex);

    // TODO: Добавить проверку на порог
    return int(maxIndex);
}

// Точка входа
int main(void)
{
    cv::Mat img = cv::imread("./dataset/img_scan500.png");

    std::vector<cv::Mat> processed;
    prepareImage(img, processed);

    cv::Size geometry = img.size();
    cv::Mat resImage = cv::Mat(geometry.height, geometry.width, CV_8UC3, cv::Scalar(255, 255, 255));

    const cv::Mat &preparedImg = processed.back();

    for (int x = 0; x < geometry.width; x++)
    {
        int y = f(preparedImg, x);

        cv::Vec3b &rgb = resImage.at<cv::Vec3b>(y, x);
        for (int i = 0; i < 3; i++)
        {
            rgb[i] = 0;
        }
    }

    // cv::Mat images[] = {img, preparedImg, blankImage};
    cv::Mat images[] = {img, resImage};
    cv::Mat res;
    cv::vconcat(images, sizeof(images) / sizeof(images[0]), res);
    cv::imshow("res", res);

    cv::waitKey(0);
    return 0;
}