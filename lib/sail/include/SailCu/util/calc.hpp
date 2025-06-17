#include <cmath>

namespace sail {
inline float radians(float degree) {
	return degree * 0.017453292519943295769236907684886f;
}

template<typename Vec3T>
Vec3T cross(Vec3T v1, Vec3T v2) {
	return Vec3T(
		v1[1] * v2[2] - v1[2] * v2[1],
		v1[2] * v2[0] - v1[0] * v2[2],
		v1[0] * v2[1] - v1[1] * v2[0]);
}

template<typename Vec3T>
Vec3T normalize(Vec3T v) {
	float length = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	if (length > 0.0f) {
		v /= length;
	}
	return v;
}
}// namespace sail
