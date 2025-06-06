#pragma once

#include "Ray.hpp"
#include "CollisionData.hpp"

class Shape {
public:
    virtual bool collide(const Ray& ray, float t_min, float t_max, CollisionData& cd) const = 0;
};
