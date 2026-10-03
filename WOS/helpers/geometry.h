#pragma once

// Right now it is only defined for 2D simple geometry (build using connected segments, circles and arcs)

#include <cmath>
#include <complex>
#include <array>
#include <algorithm>
#include <vector>
#include <limits>
#include <numbers>
#include <cstddef>

inline constexpr double pi = std::numbers::pi;

using Vec2D = std::complex<float>;
inline float dot(Vec2D u, Vec2D v) { return std::real(std::conj(u) * v); }
inline float cross2D(const Vec2D u, const Vec2D v) { return u.real() * v.imag() - u.imag() * v.real(); }
inline float length(Vec2D u) { return sqrt(norm(u)); }

// Basic domain building blocks
// Check that segment start point and end point isnt the same.
using Segment = std::array<Vec2D, 2>;

struct Circle
{
    Vec2D center;
    float radius;
};

struct Arc
{
    Vec2D center;
    float startAngle;
    float endAngle;
    float radius;
};

// returns the point on segment s closest to x
Vec2D closestPointSegment(const Vec2D x, const Segment s);
Vec2D closestPointCircle(const Vec2D x, const Circle C);
Vec2D closestPointArc(const Vec2D x, const Arc A);

float distancePointSegment(const Vec2D x, const Segment s);
float distancePointCircle(const Vec2D x, const Circle C);
float distancePointArc(const Vec2D x, const Arc A);

// How do we distingish between the interior and the outside of the domain?
class Domain
{
public:
    Domain(const std::vector<Segment> &segments, const std::vector<Circle> &circles, const std::vector<Arc> &arcs)
        : m_segments(segments), m_circles(circles), m_arcs(arcs) {};

    bool isPointInDomain(Vec2D x) const;
    bool isPointInBoundary(Vec2D x) const;
    float distanceBoundary(Vec2D x) const;

private:
    // Should own memory or how do i need to handle it?
    std::vector<Segment> m_segments;
    std::vector<Circle> m_circles;
    std::vector<Arc> m_arcs;
};