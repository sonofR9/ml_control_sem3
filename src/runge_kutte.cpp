#include <functional>
#include <iostream>

constexpr double kEps = 1e-10;

double RungeKutte(double startT, double startX, std::function<double(double, double)> fun, double interestT,
                  double delta = 0.001)
{
    double curT{startT};
    double curX{startX};
    while (curT < interestT)
    {
        double k1 = fun(curX, curT);
        double k2 = fun(curX + delta / 2 * k1, curT + delta / 2);
        double k3 = fun(curX + delta / 2 * k2, curT + delta / 2);
        double k4 = fun(curX + delta * k3, curT + delta);
        curX += delta / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        curT += delta;
    }

    return curX;
}

int main()
{
    auto fun = [](double /*x*/, double t) -> double { return t * t; };

    double delta{1};
    double curX{1};
    double curT{0};
    double endT{1};
    std::cout << curX << std::endl;
    while (curT < endT - kEps)
    {
        curX = RungeKutte(curT, curX, fun, curT + delta, delta);
        curT += delta;
        std::cout << curX << std::endl;
    }
    return 0;
}
