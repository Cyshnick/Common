#ifndef CONTRCALCULATORBACK_H
#define CONTRCALCULATORBACK_H

#include "iContrAlgMat.h"
#include "cvsharedlibtempl.h"
#include "ContrUnitBack_global.h"

class CONTRUNITBACK_EXPORT ContrCalculatorBack : public IContrCalculatorExt
{
public:
    ContrCalculatorBack(Settings const& settings);

    ContrCalculatorBack(ContrCalculatorBack const&) = default;
    ContrCalculatorBack(ContrCalculatorBack &&) = default;

    virtual void setSettings(const Settings &setts) override;
    virtual void setImage(const cv::Mat &image) override;
    virtual CalculatorOutput getResult() override;
    virtual void calc() override;

private:
    void clearAllStorages();
    void initValuesForCalculation();
    void shrinkImageAccountStepSize();
    void checkSettingsOnAdequacy() const;
    void normAveragedLine();
    void setAlignLinesItemsToNull();
    void putAveragedDataToResult();
    void calcContrast();
    void calcDarkLightPoints();
    cv::Rect calcRectOfAveragedDataIndice(size_t i);

    std::vector<double> alignData(const std::vector<double> &vect);

    template<typename T>
    void execOperationOnImage() {
        calcMeanAndStdDeviation<T>();
        calcAveragedData<T>();
    }

    template<typename T>
    void calcMeanAndStdDeviation() {
        double mean = 0;
        double stdDev = 0;

        for(auto itPixel = processed.begin<T>();
            itPixel < processed.end<T>();
            itPixel++)
        {
            mean += *itPixel;
            stdDev += static_cast<double>(*itPixel)*(*itPixel);
        }

        mean /= numOfPixels;
        result.stdDeviation = sqrt(stdDev/numOfPixels - mean*mean);
        result.averagedIntensity = mean;
    }

    template<typename T>
    void calcAveragedData() {
        size_t stepLineOffset = processed.cols*settings.heightStep;
        for(size_t k = 0; k < numOfPixels; k+= stepLineOffset)
            calcFullAveragedLine<T>(k);

        if(settings.intensityAlignEnabled)
            averagedData = alignData(averagedData);

        putAveragedDataToResult();
    }

    template<typename T>
    void calcFullAveragedLine(size_t startPixelOfLineIndice) {
        calcAveragedLine(processed.begin<T>() + startPixelOfLineIndice);

        if(settings.intensityAlignEnabled)
            averagedLine = alignData(averagedLine);

        averagedData.insert(averagedData.end(), averagedLine.begin(), averagedLine.end());

        setAlignLinesItemsToNull();
    }

    template <typename It>
    void calcAveragedLine(It iterator) {
        double mean = 0;
        for(size_t i = 0; i < settings.heightStep; i++)
            for(size_t j = 0; j < numOfHorizontalSteps; j++)
            {
                for(size_t k = 0; k < settings.widthStep; k++, iterator++)
                    mean += *iterator;

                averagedLine.at(j) += mean;
                mean = 0;
            }

        normAveragedLine();
    }

    Settings settings;
    CalculatorOutput result;
    cv::Mat source;
    cv::Mat processed;
    std::vector<double> averagedLine;
    std::vector<double> averagedData;
    std::vector<double> averagedDataSorted;

    size_t numOfHorizontalSteps;
    size_t numOfVerticalSteps;
    size_t numOfPixels;
    size_t numOfPixelsInStep;
};



Q_DECL_EXPORT IContrCalculatorExt *buildContrCalculatorExt(Settings const&);

#endif // CONTRCALCULATORBACK_H
