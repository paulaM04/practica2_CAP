#pragma once

#include "Vec3.hpp"

void writeBMP(const char* filename, unsigned char* data, int w, int h);

float schlick(float cosine, float ref_idx);
bool refract(const Vec3& v, const Vec3& n, float ni_over_nt, Vec3& refracted);
Vec3 reflect(const Vec3& v, const Vec3& n);
