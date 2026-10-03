// Based on Keenan Crane code found in
//      https://www.cs.cmu.edu/~kmcrane/Projects/MonteCarloGeometryProcessing/WoSPoisson2D.cpp.html

// Quick and dirty implementation of a 2D Poisson solver via random walks.
// Corresponds to the naïve estimator given in Equation 8 of
// Sawhney & Crane, "Monte Carlo Geometry Processing" (2020).
// To compile: c++ -std=c++20 -fopenmp -O3 -pedantic -Wall main.cpp -o poisson
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <omp.h>

#include "helpers/geometry.h"
using namespace std;

// thread_local std::mt19937 rng{std::random_device{}()};
static thread_local std::mt19937 rng{1234567 + 7919u * (unsigned)omp_get_thread_num()};

static float random(float rMin, float rMax)
{
    std::uniform_real_distribution<float> d(rMin, rMax);
    return d(rng);
}

// harmonic Green's function for a 2D ball of radius R
float G(float r, float R)
{
    float GrR = log(R / r) / (2. * pi);
    if (isnan(GrR))
        return 0;
    return GrR;
}

// solves a Laplace equation Δu = f at x0, where the boundary is given
// by a collection of segments, and the boundary conditions are given
// by a function g that can be evaluated at any point in space
float solve(Vec2D x0,                        // evaluation point
            const Domain &dom,               // Domain where the edp is defined
            const function<float(Vec2D)> &f, // source term
            const function<float(Vec2D)> &g  // boundary conditions
)
{
    const float eps = 0.001; // stopping tolerance
    const int nWalks = 256;  // number of Monte Carlo samples
    const int maxSteps = 64; // maximum walk length

    float sum = 0.;
    for (int i = 0; i < nWalks; i++)
    {
        Vec2D x = x0;
        float R;
        int steps = 0;
        do
        {
            R = dom.distanceBoundary(x);

            // sample a point y uniformly from the ball of radius R around x
            float r = R * sqrt(random(0., 1.));
            float alpha = random(0., 2. * pi);
            Vec2D y = x + Vec2D(r * cos(alpha), r * sin(alpha));
            sum += (pi * R * R) * f(y) * G(r, R);

            // sample the next point x uniformly from the sphere around x
            float theta = random(0., 2. * pi);
            x = x + Vec2D(R * cos(theta), R * sin(theta));
            steps++;
        } while (R > eps && steps < maxSteps);

        // Right now we add the value at the point x even if it has not reach the boundary. How could we fix this?
        sum += g(x);
    }

    return sum / nWalks; // Monte Carlo estimate
}

// reference solution
float uref(Vec2D x)
{
    return cos(2.f * pi * real(x)) * sin(2.f * pi * imag(x));
}

// Laplacian of reference solution
float laplace_uref(Vec2D x)
{
    return -8.f * (pi * pi) * cos(2.f * pi * real(x)) * sin(2.f * pi * imag(x));
}

// four segments enclosing the unit square
static vector<Segment> scene = {
    {{Vec2D(0.0, 0.0), Vec2D(1.0, 0.0)}},
    {{Vec2D(1.0, 0.0), Vec2D(1.0, 1.0)}},
    {{Vec2D(1.0, 1.0), Vec2D(0.0, 1.0)}},
    {{Vec2D(0.0, 1.0), Vec2D(0.0, 0.0)}}};

void poisson()
{
    bool save = false;

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
    // To validate the implementation we solve the Poisson equation
    //
    //    Δu = Δu0  on Ω
    //     u =  u0  on ∂Ω
    //
    // where u0 is some reference function.  The solution should
    // converge to u = u0 as the number of samples N increases.

    // Should we explicity call new?
    auto circles = std::vector<Circle>{};
    auto arcs = std::vector<Arc>{};
    Domain domain = Domain(scene, circles, arcs);

    float error{};

    const int s = 128; // Resolution of the geometry
    std::array<float, s * s> solution{};

#pragma omp parallel for reduction(+ : error) schedule(static)
    for (int j = 0; j < s; j++)
    {
        // cerr << "row " << j << " of " << s << endl;
        for (int i = 0; i < s; i++)
        {
            Vec2D x0((float)i / (float)s, (float)j / (float)s);
            float u = solve(x0, domain, laplace_uref, uref);
            if (save)
                solution[j * s + i] = u;

            // Compute error against solution
            error += (uref(x0) - u) * (uref(x0) - u);
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

    std::cout << "The error against the solution is: " << sqrt(error / (s * s)) << std::endl;

    // return 0;
}
