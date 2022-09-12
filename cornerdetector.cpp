#include "cornerdetector.h"


namespace corner_detector
{

using namespace cv;
using namespace std;

double calcDist(const cv::Point &a, const cv::Point &b)
{
    double dx = b.x - a.x;
    double dy = b.y - a.y;
    return sqrt(dx*dx + dy*dy);
}


template<typename T, typename V>
void polyLineSplit(const std::vector<cv::Point_<T> >& polyline,
                   std::vector<V>& contourx,
                   std::vector<V>& contoury)
{
    contourx.resize(polyline.size());
    contoury.resize(polyline.size());

    for (size_t j=0; j < polyline.size(); j++)
    {
        contourx[j] = (V)(polyline[j].x);
        contoury[j] = (V)(polyline[j].y);
    }
}

template<typename T, typename V>
void PolyLineMerge(std::vector<cv::Point_<T> >& polyline,
                   const std::vector<V>& contourx,
                   const std::vector<V>& contoury)
{
    assert(contourx.size()==contoury.size());

    polyline.resize(contourx.size());
    for (size_t j=0; j < contourx.size(); j++)
    {
        polyline[j].x = static_cast<T>(contourx[j]);
        polyline[j].y = static_cast<T>(contoury[j]);
    }
}

void findContourCorners(const Points &contour, Points &resCont)
{
    double sigma = baseGaussSigma/baseKernelSize*contour.size();

    int kernalSize = round((6.*sigma+1.)/2.)*2 + 1;

    vector<double> gauss, dGauss, d2Gauss;
    getGaussianDerivs(sigma, kernalSize, gauss, dGauss, d2Gauss);

    vector<double> curvex, curvey, smoothx, smoothy;
    polyLineSplit(contour, curvex, curvey);

    vector<double> smoothDx, smoothD2x, smoothDy, smoothD2y;
    getDxCurve(curvex, smoothx, smoothDx, smoothD2x, gauss, dGauss, d2Gauss);
    getDxCurve(curvey, smoothy, smoothDy, smoothD2y, gauss, dGauss, d2Gauss);

    vector<double> kappa(curvex.size());
    for(size_t i = 0; i < curvex.size(); i++)
        kappa[i] = (smoothDx[i]*smoothD2y[i] - smoothD2x[i]*smoothDy[i]) / pow(smoothDx[i]*smoothDx[i] + smoothDy[i]*smoothDy[i], 1.5);

    vector<int> cornerIndices;
    findKappaExtr(kappa, cornerIndices);

    Points smoothCont;
    PolyLineMerge(smoothCont, smoothx, smoothy);

    Points primaryCorners;
    for(auto i : cornerIndices)
        primaryCorners.push_back(smoothCont[i]);

    dropExcessPoints(primaryCorners, resCont);
}

void getGaussianDerivs(double sigma, int M, Doubles &gaussian, Doubles &dg, Doubles &d2g)
{
    int L = (M - 1)/ 2;
    double sigma_sq = sigma*sigma;
    double sigma_quad = sigma_sq*sigma_sq;
    dg.resize(M); d2g.resize(M); gaussian.resize(M);

    Mat_<double> g = getGaussianKernel(M, sigma, CV_64F);
    for(double i = -L; i < L+1.; i += 1.)
    {
        int idx = static_cast<int>(i + L);
        gaussian[idx] = g(idx);
        dg[idx] = (-i/sigma_sq)*g(idx);
        d2g[idx] = (-sigma_sq + i*i)/sigma_quad*g(idx);
    }
}

void getdX(Doubles const& x, int n, double &gx, double &dgx, double &d2gx,
           Doubles const& g, Doubles const& dg, Doubles const& d2g)
{
    size_t L = (g.size() - 1) / 2;
    gx = dgx = d2gx= 0.;
    for(size_t k = -L; k < L + 1; k++)
    {
        double x_n_k;
        if(n-k < 0)
            x_n_k = x[x.size() + (n - k)];
        else if(n-k > x.size() - 1)
            x_n_k = x[(n-k) - (x.size())];
        else
            x_n_k = x[n-k];

        gx += x_n_k*g[k + L];
        dgx += x_n_k*dg[k + L];
        d2gx += x_n_k*d2g[k + L];
    }
}

void getDxCurve(Doubles const& x, Doubles &gx, Doubles &dx,
                Doubles &d2x, Doubles const& g, Doubles const& dg, Doubles const& d2g)
{
    gx.resize(x.size());
    dx.resize(x.size());
    d2x.resize(x.size());
    for(size_t i = 0; i < x.size(); i++)
    {
        double gausx, dgx, d2gx;
        getdX(x, i, gausx, dgx, d2gx, g, dg, d2g);
        gx[i] = gausx;
        dx[i] = dgx;
        d2x[i] = d2gx;
    }
}

void findKappaExtr(Doubles const& kappa, std::vector<int> &corners)
{
    double f0 = kappa.back();
    vector<double> extr;
    vector<int> extrIndex;
    for(size_t i = 0; i < kappa.size(); i++)
    {
        double f1 = kappa[i];

        double f2;
        if(i + 1 != kappa.size())
            f2 = kappa[i+1];
        else
            f2 = kappa.front();

        if((fabs(f0) < fabs(f1))&&(fabs(f1) > fabs(f2)))
        {
            extr.push_back(f1);
            extrIndex.push_back(i);
        }
        f0 = f1;
    }

    double mean = 0.;
    for(auto ex : extr)
        mean += fabs(ex);
    mean /= extr.size();

    for(size_t i = 0; i < extr.size(); i++)
        if(fabs(extr[i]) > privarlyThreshKappa*mean)
            corners.push_back(extrIndex[i]);
}

void dropExcessPoints(const Points &cont, Points &resCont)
{
    double meanDistBetweenCorners = 0.;
    auto prev = cont.back();
    for(auto const& cur : cont)
    {
        meanDistBetweenCorners += calcDist(cur, prev);
        prev = cur;
    }
    meanDistBetweenCorners /= cont.size();

    resCont.clear();
    prev = cont.back();
    for(auto cur = cont.begin(); cur < cont.end(); cur++)
    {
        double distToPrevCorner = calcDist(prev, *cur);

        Point next;
        if(cur + 1 == cont.end())
            next = cont.front();
        else
            next = cur[1];

        double distToNextCorner = calcDist(next, *cur);

        if(distToPrevCorner + distToNextCorner > finallyThreshCorners*meanDistBetweenCorners)
        {
            resCont.push_back(*cur);
            prev = *cur;
        }
    }
}

}
