#ifndef CORNERDETECTOR_H
#define CORNERDETECTOR_H

#include <opencv2/opencv.hpp>

using Points = std::vector<cv::Point>;
using Doubles = std::vector<double>;

namespace corner_detector
{

const double finallyThreshCorners = 1.4;
const double privarlyThreshKappa = 0.5;
const double baseGaussSigma = 8.;
const double baseKernelSize = 6900;


double calcDist(cv::Point const& a, cv::Point const& b);

void findContourCorners(const Points &contour, Points &resCont);

void getGaussianDerivs(double sigma, int M, Doubles &gaussian, Doubles &dg, Doubles &d2g);

void getdX(Doubles const& x, int n, double& gx, double& dgx, double& d2gx, Doubles const& g,
           Doubles const& dg, const Doubles& d2g);

void getDxCurve(Doubles const& x, Doubles &gx, Doubles &dx, Doubles &d2x, Doubles const& g,
                Doubles const& dg, Doubles const& d2g);

void findKappaExtr(Doubles const& kappa, std::vector<int>& corners);

void dropExcessPoints(Points const& cont, Points& resCont);


}

#endif // CORNERDETECTOR_H
