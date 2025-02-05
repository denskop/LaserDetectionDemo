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

#include <Eigen/Dense>
#include <unsupported/Eigen/LevenbergMarquardt>

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
    cv::Size geometry = channel.size();

    // Копирование пикселей в новый вектор с расширением типа данных
    // uchar -> uint16_t
    size_t pixNum = channel.total();
    std::vector<uint16_t> pix(pixNum);

    // Возведение в квадрат
    for (int i = 0; i < pixNum; i++)
    {
        pix[i] = pow(channel.at<uchar>(i), 2);
    }

    // Нормализация значений
    uint16_t max = *std::max_element(pix.begin(), pix.end());
    uint16_t min = *std::min_element(pix.begin(), pix.end());
    uint16_t delta = max - min > 0 ? max - min : 1;

        for (auto i = pix.begin(); i < pix.end(); i++)
    {
        *i = 255 * (*i - min) / delta;
    }

    // Создание и заполнение внутреннего буфера матрицы
    uchar *buffer = new uchar[pixNum];
    for (int i = 0; i < pixNum; i++)
    {
        buffer[i] = uchar(pix[i]);
    }

    return cv::Mat(geometry, CV_8U, buffer);
}

cv::Mat applyPixSqrWithNorm(const cv::Mat &img)
{
    cv::Mat channels[3];
    cv::split(img, channels);

cv::Mat resChannels[3];
    resChannels[0] = applyPixSqrWithNormCh(channels[0]);
    for (int i = 1; i < 3; i++)
    {
        resChannels[i] = resChannels[0];
    }

cv::Mat res;
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
    cv::GaussianBlur(img, res, cv::Size(1, 11), 0);
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
    cv::inRange(img, cv::Scalar(90, 20, 20), cv::Scalar(255, 255, 255), mask);
    cv::bitwise_and(img, img, res, mask);
    return res;
}

cv::Mat erodeDilateProcessor(const cv::Mat &img)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 1));
    cv::Mat resEroded, resDilated;
    cv::erode(img, resEroded, kernel);
    cv::dilate(resEroded, resDilated, kernel);
    cv::Size geometry = img.size();
    return resDilated;
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

    auto a = Eigen::VectorXf::NullaryExpr(norm.size(), [&norm](Eigen::Index i)
                                          { return i * pow(norm(i), 2); });
    auto b = norm.unaryExpr([](float x)
                            { return pow(x, 2); });
    return a.sum() / b.sum();
}

float gauss(float x, float a, float mu, float sigma)
{
    return a * exp(-0.5 * powf((x - mu) / sigma, 2));
}

struct GaussSolver : Eigen::DenseFunctor<double>
{
    Eigen::VectorXd x, y;

    GaussSolver(const Eigen::MatrixX2d &f_variables) : DenseFunctor<double>(
                                                           3,
                                                           f_variables.rows()),
                                                       x(f_variables.col(0)),
                                                       y(f_variables.col(1))
    {
    }

    int operator()(const InputType &coeffs, ValueType &f_values)
    {
        double a = coeffs[0];
        double mu = coeffs[1];
        double sigma = coeffs[2];

        auto power = -(x - ValueType::Constant(values(), mu)).array() / sigma;
        f_values = a * (-0.5 * power.square()).exp() - y.array();
        return 0;
    }

    int df(const InputType &coeffs, JacobianType &f_values)
    {
        double a = coeffs[0];
        double mu = coeffs[1];
        double sigma = coeffs[2];

        auto muVector = ValueType::Constant(values(), mu);

        auto j0 = (-0.5 * ((x - muVector).array() / sigma).square()).exp();
        auto j1 = a * (x - muVector).array() / pow(sigma, 2) * j0;

        f_values.col(0) = j0;
        f_values.col(1) = j1;
        f_values.col(2) = (x - muVector).array() * j1 / sigma;

        return 0;
    }
};

float calcIntenseAccurate(const Eigen::VectorXi &in)
{
    std::vector<double> test;

    for (auto p : in)
    {
        test.push_back(p);
    }

    auto x = Eigen::VectorXd::LinSpaced(220, 0, 219);
    auto y = Eigen::Map<Eigen::VectorXd>(test.data(), test.size());

    Eigen::MatrixX2d f(x.size(), 2);
    f << x, y;

    GaussSolver gaussSolver(f);

    int ixMax;
    double a = in.maxCoeff(&ixMax);
    double mu = in[ixMax];
    auto deviationVector = in.unaryExpr([mu](int x)
                                        { return pow(x - mu, 2); });
    double sigma = sqrt(deviationVector.mean());

    Eigen::VectorXd params(3);
    params << a, mu, sigma;

    Eigen::LevenbergMarquardt<GaussSolver> solver(gaussSolver);

    solver.setXtol(1.0e-6);
    solver.setFtol(1.0e-6);
    solver.minimize(params);

    a = params[0];
    mu = params[1];
    sigma = params[2];

    return gauss(mu, a, mu, sigma);
}

// I = f(x)
int f(const cv::Mat &img, int x, float intense_delta_thr = 0.5)
{
    cv::Mat vert_line = img.col(x);
    cv::Mat channels[3];
    cv::split(vert_line, channels);

    Eigen::VectorXi redChannel;
    cv::cv2eigen(channels[0], redChannel);

    Eigen::VectorXf normChannel;
    calcNorm(redChannel, normChannel);
    float max = normChannel.maxCoeff();
    float mean = normChannel.mean();

    // Расчет центра масс
    int res = calcMassCenter(redChannel);
    return max - mean > intense_delta_thr ? res : -1;
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

        if (y < 0)
        {
            continue;
        }

        cv::Vec3b &rgb = resImage.at<cv::Vec3b>(y, x);
        for (int i = 0; i < 3; i++)
        {
            rgb[i] = 0;
        }
    }

    cv::Mat images[] = {img, preparedImg, resImage};
    cv::Mat res;
    cv::vconcat(images, sizeof(images) / sizeof(images[0]), res);

    cv::imshow("res", res);

    cv::waitKey(0);
    return 0;
}