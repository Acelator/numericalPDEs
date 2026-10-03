#include "geometry.h"

Vec2D closestPointSegment(const Vec2D x, const Segment s)
{
    Vec2D u = s[1] - s[0];
    float t = std::clamp(dot(x - s[0], u) / dot(u, u), 0.f, 1.f);
    return (1 - t) * s[0] + t * s[1];
}

Vec2D closestPointCircle(const Vec2D x, const Circle C)
{
    // Every point is the closest one. We just pick one of them.
    if (x == C.center)
        return C.center + Vec2D(C.radius, 0.f);

    Vec2D z = (x - C.center) / length(x - C.center);
    return C.center + z * C.radius;
}

Vec2D closestPointArc(const Vec2D x, const Arc A)
{
    // Every point is the closest one. We just pick one of them.
    if (x == A.center)
        return A.center + Vec2D(A.radius, 0.f);

    Vec2D u = Vec2D(cos(A.startAngle), sin(A.startAngle));
    Vec2D v = Vec2D(cos(A.endAngle), sin(A.endAngle));
    Vec2D w = x - A.center;

    float phi = atan2(static_cast<double>(cross2D(u, w)), dot(u, w));

    if (phi < 0.)
        phi += 2. * pi;

    if (phi < A.endAngle - A.startAngle)
    {
        Vec2D unitVector = w / length(w);
        return A.center + unitVector * A.radius;
    }

    Vec2D a = A.center + A.radius * u;
    Vec2D b = A.center + A.radius * v;

    return std::min(length(x - a), length(x - b)) == length(x - a) ? a : b;
}

float distancePointSegment(const Vec2D x, const Segment s)
{
    Vec2D y = closestPointSegment(x, s);
    return length(x - y);
}

float distancePointCircle(const Vec2D x, const Circle C)
{
    Vec2D y = closestPointCircle(x, C);
    return length(x - y);
}

float distancePointArc(const Vec2D x, const Arc A)
{
    Vec2D y = closestPointArc(x, A);
    return length(x - y);
}

// Domain member functions

bool Domain::isPointInDomain(Vec2D x) const
{
    return false;
}

bool Domain::isPointInBoundary(Vec2D x) const
{
    return false;
}

// ! Make more efficient. Research ways to restrict the checks only to a smaller subregion.
//      Ideally I wont need to check against every block of the domain.
float Domain::distanceBoundary(Vec2D x) const
{
    if (m_segments.empty() && m_circles.empty() && m_arcs.empty())
        return 0.;

    // get the distance to the closest point on any segment
    float R = std::numeric_limits<float>::max();
    for (const auto &s : this->m_segments)
    {
        R = std::min(R, distancePointSegment(x, s));
    }

    for (const auto &c : this->m_circles)
    {
        R = std::min(R, distancePointCircle(x, c));
    }

    for (const auto &a : this->m_arcs)
    {
        R = std::min(R, distancePointArc(x, a));
    }

    return R;
}