#include "Vec3.hpp"

Vec3::Vec3() : e{0, 0, 0} {}
Vec3::Vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}

const Vec3& Vec3::operator+() const { return *this; }
Vec3 Vec3::operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
// float Vec3::operator[](int i) const { return e[i]; }
float& Vec3::operator[](int i) { return e[i]; }
const float& Vec3::operator[](int i) const { return e[i]; }

float Vec3::x() const { return e[0]; }
float Vec3::y() const { return e[1]; }
float Vec3::z() const { return e[2]; }
float Vec3::r() const { return e[0]; }
float Vec3::g() const { return e[1]; }
float Vec3::b() const { return e[2]; }
    
float Vec3::length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
float Vec3::squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
void Vec3::make_unit_vector() { *this /= length(); }

Vec3& Vec3::operator+=(const Vec3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

Vec3& Vec3::operator*=(const Vec3& v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

Vec3& Vec3::operator/=(const Vec3& v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

Vec3& Vec3::operator-=(const Vec3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

Vec3& Vec3::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

Vec3& Vec3::operator/=(const float t) {
    float k = 1.0f / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

/*****************************************************************************/
/* Extern operators Functions                                                */
/*****************************************************************************/

Vec3 operator+(Vec3 v1, const Vec3& v2) {
    v1 += v2;
    return (v1);
}

Vec3 operator-(Vec3 v1, const Vec3& v2) {
    v1 -= v2;
    return (v1);
}

Vec3 operator*(Vec3 v1, const Vec3& v2) {
    v1 *= v2;
    return (v1);
}

Vec3 operator/(Vec3 v1, const Vec3& v2) {
    v1 /= v2;
    return (v1);
}

Vec3 operator*(float t, Vec3 v) {
    v *= t;
    return (v);
}

Vec3 operator/(Vec3 v, float t) {
    v /= t;
    return (v);
}

Vec3 operator*(Vec3 v, float t) {
    v *= t;
    return (v);
}

/*****************************************************************************/
/* Friend Functions                                                          */
/*****************************************************************************/

float dot(const Vec3& v1, const Vec3& v2) {
    return v1.e[0] * v2.e[0]
        + v1.e[1] * v2.e[1]
        + v1.e[2] * v2.e[2];
}

Vec3 cross(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1],
        v1.e[2] * v2.e[0] - v1.e[0] * v2.e[2],
        v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]);
}

/*****************************************************************************/
/* Other Functions                                                          */
/*****************************************************************************/

Vec3 unit_vector(Vec3 v) {
    return v / v.length();
}

std::istream& operator>>(std::istream& is, Vec3& t) {
    is >> t[0] >> t[1] >> t[2];
    return is;
}

std::ostream& operator<<(std::ostream& os, const Vec3& t) {
    os << t[0] << " " << t[1] << " " << t[2];
    return os;
}
