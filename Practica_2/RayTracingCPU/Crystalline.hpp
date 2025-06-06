#pragma once

#include "Utils.hpp"
#include "Random.hpp"

#include "Material.hpp"

class Crystalline : public Material {
public:
    Crystalline(float ri) : ref_idx(ri) {}

    bool scatter(const Ray& r_in, const CollisionData& cd, Vec3& attenuation, Ray& scattered) const;

private:
    float ref_idx;
};

