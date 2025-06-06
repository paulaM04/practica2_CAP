#pragma once

/*****************************************************************************/
/* Based on the code written in 2016 by Peter Shirley <ptrshrl@gmail.com>    */
/* Check COPYING.txt for copyright license                                   */
/*****************************************************************************/

#include <cmath>
#include <iostream>

class Vec3 {
public:
    Vec3();
    Vec3(float e0, float e1, float e2);

    const Vec3& operator+() const;
    Vec3 operator-() const;
    // float operator[](int i) const;
    float& operator[](int i);
    const float& operator[](int i) const;

    Vec3& operator+=(const Vec3& v);
    Vec3& operator*=(const Vec3& v);
    Vec3& operator/=(const Vec3& v);
    Vec3& operator-=(const Vec3& v);
    Vec3& operator*=(const float t);
    Vec3& operator/=(const float t);

    float x() const;
    float y() const;
    float z() const;
    float r() const;
    float g() const;
    float b() const;
        
    float length() const;
    float squared_length() const;
    void make_unit_vector();

    friend float dot(const Vec3& v1, const Vec3& v2);
    friend Vec3 cross(const Vec3& v1, const Vec3& v2);
private:
    float e[3];
};

/*****************************************************************************/
/* Extern operators Functions                                                */
/*****************************************************************************/

Vec3 operator+(Vec3 v1, const Vec3& v2);
Vec3 operator-(Vec3 v1, const Vec3& v2);
Vec3 operator*(Vec3 v1, const Vec3& v2);
Vec3 operator/(Vec3 v1, const Vec3& v2);
Vec3 operator*(float t, Vec3 v);
Vec3 operator/(Vec3 v, float t);
Vec3 operator*(Vec3 v, float t);

/*****************************************************************************/
/* Friend Functions                                                          */
/*****************************************************************************/

float dot(const Vec3& v1, const Vec3& v2);

Vec3 cross(const Vec3& v1, const Vec3& v2);

/*****************************************************************************/
/* Other Functions                                                          */
/*****************************************************************************/

Vec3 unit_vector(Vec3 v);

std::istream& operator>>(std::istream& is, Vec3& t);
std::ostream& operator<<(std::ostream& os, const Vec3& t);
