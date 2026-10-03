// Based on Keenan Crane code found in
//      https://www.cs.cmu.edu/~kmcrane/Projects/MonteCarloGeometryProcessing/WoSLaplace2D.cpp.html
// (Slow) implementation of Muller's 1956 "Walk on Spheres" algorithm
// Corresponds to the naïve estimator given in Equation 5 of
// Sawhney & Crane, "Monte Carlo Geometry Processing" (2020).
// To compile: c++ -std=c++20 -O3 -fopenmp -pedantic -Wall main.cpp -o wos

#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <limits>
#include <fstream>
#include <omp.h>
using namespace std;

#include "WoSLaplace2D.h"

// thread_local std::mt19937 rng{std::random_device{}()};
static thread_local std::mt19937 rng{123456 + 7919u * (unsigned)omp_get_thread_num()};

// returns a random value in the range [rMin,rMax]
static float random(float rMin, float rMax)
{
    std::uniform_real_distribution<float> d(rMin, rMax);
    return d(rng);
}

// solves a Laplace equation Δu = 0 at x0, where the boundary is given
// and the boundary conditions are given by a function g that can be evaluated at any point in space.
float solve(const Vec2D &x0, const Domain &dom, const function<float(Vec2D)> &g)
{
    const float eps = 0.01;  // stopping tolerance
    const int nWalks = 256;  // number of Monte Carlo samples
    const int maxSteps = 64; // maximum walk length

    float sum = 0.;
    for (int i = 0; i < nWalks; i++)
    {
        Vec2D x = x0;
        float R{};
        int steps = 0;
        do
        {
            R = dom.distanceBoundary(x);
            float theta = random(0., 2. * pi);
            x = x + Vec2D(R * cos(theta), R * sin(theta)); // Point sampled in the new sphere with radius R around x
            steps++;
        } while (R > eps && steps < maxSteps);

        sum += g(x);
    }
    return sum / nWalks; // Monte Carlo estimate
}

float checker(Vec2D x)
{
    const float s = 6.;
    return fmod(floor(s * real(x)) + floor(s * imag(x)), 2.);
}

static vector<Segment> scene = {
    {{Vec2D(0.5, 0.1), Vec2D(0.9, 0.5)}},
    {{Vec2D(0.5, 0.9), Vec2D(0.1, 0.5)}},
    {{Vec2D(0.1, 0.5), Vec2D(0.5, 0.1)}},
    {{Vec2D(0.5, 0.33333333), Vec2D(0.5, 0.6666666)}},
    {{Vec2D(0.33333333, 0.5), Vec2D(0.6666666, 0.5)}}};

// Solves and store the solution to a file
void laplace()
{
    bool save = true;

    ofstream out;
    if (save)
    {
        out.open("out.csv");
        if (!out)
        {
            std::cerr << "failed to open out.csv\n";
            // return 1;
        }
    }

    const int s = 128; // image size

    auto circles = std::vector<Circle>{};
    auto arcs = std::vector<Arc>{};
    Domain domain = Domain(scene, circles, arcs);
    std::array<float, s * s> solution{};
    float error{};

#pragma omp parallel for reduction(+ : error) schedule(static)
    for (int j = 0; j < s; j++)
    {
        // cerr << "row " << j << " of " << s << endl;
        for (int i = 0; i < s; i++)
        {
            Vec2D x0((float)i / (float)s, (float)j / (float)s);
            float u = solve(x0, domain, checker);
            if (save)
                solution[j * s + i] = u;

            // Compute error against solution
            // error += (uref(x0) - u) * (uref(x0) - u);
        }
    }

    if (save)
    {
        for (int j = 0; j < s; j++)
        {
            for (int i = 0; i < s; i++)
            {
                out << solution[j * s + i];
                if (i < s - 1)
                    out << ",";
            }
            out << endl;
        }
    }

    // std::cout << "The error against the solution is: " << sqrt(error) << std::endl;
    // return 0;
}
