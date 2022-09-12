#include "contrcalculatorback.h"

namespace cvShared {
    extern "C++" void saveImage(cv::Mat const&src, std::string const& pref);
}

using namespace cv;

ContrCalculatorBack::ContrCalculatorBack(Settings const& settings) :
    settings(settings)
{
    checkSettingsOnAdequacy();
}

void ContrCalculatorBack::checkSettingsOnAdequacy() const
{
    if(settings.heightStep == 0 || settings.widthStep == 0)
        throw std::runtime_error("A or B equals to zero!");

    if(settings.numberOfDarkPoints == 0 || settings.numberOfLightPoints == 0)
        throw std::runtime_error("X or Y equals to zero!");
}

void ContrCalculatorBack::setSettings(const Settings &setts)
{
    settings = setts;
    checkSettingsOnAdequacy();
}

void ContrCalculatorBack::setImage(const cv::Mat &image)
{
    source = image;
}

CalculatorOutput ContrCalculatorBack::getResult()
{
    return result;
}

void ContrCalculatorBack::calc()
{
    clearAllStorages();
    initValuesForCalculation();
    shrinkImageAccountStepSize();

    if(processed.type() == CV_8UC1)
        execOperationOnImage<uint8_t>();
    else if(processed.type() == CV_16UC1)
        execOperationOnImage<uint16_t>();
    else
        throw std::runtime_error("Wrong image format!");

    calcContrast();
    calcDarkLightPoints();
}

void ContrCalculatorBack::clearAllStorages()
{
    result.averagedData.clear();
    result.darkPoints.clear();
    result.lightPoints.clear();
    averagedLine.clear();
    averagedData.clear();
    averagedDataSorted.clear();
}

void ContrCalculatorBack::initValuesForCalculation()
{
    processed = source.clone();
    numOfHorizontalSteps = processed.cols / settings.widthStep;
    numOfVerticalSteps = processed.rows / settings.heightStep;
    numOfPixelsInStep = settings.heightStep*settings.widthStep;
    numOfPixels = numOfPixelsInStep*numOfHorizontalSteps*numOfVerticalSteps;

    averagedLine = std::vector<double>(numOfHorizontalSteps, 0);
}

void ContrCalculatorBack::shrinkImageAccountStepSize()
{
    int w = processed.cols - processed.cols % settings.widthStep;
    int h = processed.rows - processed.rows % settings.heightStep;
    processed = processed(Range(0, h), Range(0, w));
}

void ContrCalculatorBack::normAveragedLine()
{
    for(auto& step : averagedLine) {
        step /= numOfPixelsInStep;
    }
}

void ContrCalculatorBack::setAlignLinesItemsToNull()
{
    for(auto &it : averagedLine)
        it = 0;
}

void ContrCalculatorBack::putAveragedDataToResult()
{
    for(auto itAveraged = averagedData.begin(); itAveraged != averagedData.end(); itAveraged += numOfHorizontalSteps)
        result.averagedData.push_back(std::vector<double>(itAveraged, itAveraged + numOfHorizontalSteps));
}

void ContrCalculatorBack::calcContrast()
{
    averagedDataSorted = averagedData;
    std::sort(averagedDataSorted.begin(), averagedDataSorted.end());

    result.max = cvShared::calcMean(std::vector<double>(averagedDataSorted.rbegin(),
                                                        averagedDataSorted.rbegin() + settings.numberOfLightPoints));
    result.min = cvShared::calcMean(std::vector<double>(averagedDataSorted.begin(),
                                                        averagedDataSorted.begin() + settings.numberOfDarkPoints));
    if(result.max + result.min == 0)
        throw std::runtime_error("Division by zero in contrast calculator!");

    result.contrast = (result.max - result.min) / (result.max + result.min) * 100;
}

void ContrCalculatorBack::calcDarkLightPoints()
{
    auto maxEdge = *(averagedDataSorted.crbegin() + settings.numberOfLightPoints);
    auto minEdge = *(averagedDataSorted.cbegin() + settings.numberOfDarkPoints);

    auto itAveraged = averagedData.begin();
    for(size_t i = 0; itAveraged < averagedData.end(); itAveraged++, i++)
        if(*itAveraged < minEdge)
            result.darkPoints.push_back(calcRectOfAveragedDataIndice(i));
        else if(*itAveraged > maxEdge)
            result.lightPoints.push_back(calcRectOfAveragedDataIndice(i));
}

Rect ContrCalculatorBack::calcRectOfAveragedDataIndice(size_t i)
{
    int x = (i % numOfHorizontalSteps)*settings.widthStep;
    int y = (i / numOfHorizontalSteps)*settings.heightStep;
    return Rect(x, y, settings.widthStep, settings.heightStep);
}

std::vector<double> ContrCalculatorBack::alignData(const std::vector<double> &vect)
{
    std::vector<double> align = vect;
    int size = vect.size() / settings.intensityAlignCoeff - 1 + vect.size()%2;

    Mat tmp = Mat(Size(vect.size(), 1), CV_64F, align.data());
    Mat blr;
    blur(tmp, blr, Size(size, 1));
    std::vector<double> avs(blr.begin<double>(), blr.end<double>());

    double A = cvShared::calcMean(vect);
    for(auto itV = align.begin(), itA = avs.begin(); itV < align.end(); itV++, itA++){
        *itV -= *itA - A;
    }
    return align;
}


IContrCalculatorExt *buildContrCalculatorExt(Settings const& settings)
{
    return new ContrCalculatorBack(settings);
}
