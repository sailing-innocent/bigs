#pragma once
/**
 * @file camera_base.h
 * @brief The Template Camera Class
 * @author sailing-innocent
 * @date 2024-07-14
 */

#include "SailCu/util/calc.hpp"

namespace sail {

enum struct CameraCoordType : unsigned int {
	kFlipZ,
	kFlipY
};
enum struct ProjectionType : unsigned int {
	kPerspective,
	kOrthographic
};

enum struct MovementType {
	kMOVE_FORWARD,
	kMOVE_BACKWARD,
	kMOVE_LEFT,
	kMOVE_RIGHT
};

template<typename Float4x4, typename Float3, typename Float>
class Camera {
protected:
	// Core Data
	struct CameraData {
		Float3 position;
		Float3 direction;
		Float3 up;
		Float3 right;
		Float near;
		Float far;
		Float fov;// in radian
		Float aspect;
	};
	CameraData data;

public:
	explicit Camera(const CameraData& data) : data(data) {}
	explicit Camera(Float3 position, Float3 direction, Float3 up, Float3 right, Float near, Float far, Float fov, Float aspect) : data({position, direction, up, right, near, far, fov, aspect}) {}
	explicit Camera(Float3 position, Float3 target, Float3 up, Float fov, Float aspect, Float near, Float far) : data({position, normalize(target - position), up, normalize(cross(target - position, up)), near, far, fov, aspect}) {}

	Camera() = default;
	Camera(const Camera& camera) = default;
	Camera(Camera&& camera) = default;
	Camera& operator=(const Camera& camera) = default;
	Camera& operator=(Camera&& camera) = default;
	~Camera() = default;

protected:
	CameraCoordType coord_type = CameraCoordType::kFlipZ;

public:
	// get & set
	void set_coord_type(CameraCoordType type) {
		coord_type = type;
	}
	CameraCoordType get_coord_type() const {
		return coord_type;
	}

	void set_view_matrix(const Float4x4& view_matrix) {
		m_view_matrix = view_matrix;
		m_is_view_dirty = false;
	}

	void set_projection_matrix(const Float4x4& projection_matrix) {
		m_projection_matrix = projection_matrix;
		m_is_projection_dirty = false;
	}

	void set_position(const Float3 position) {
		data.position = position;
		m_is_view_dirty = true;
	}
	void move(MovementType mtype, float offset) {
		Float3 pos_offset{0.0f};
		switch (mtype) {
			case MovementType::kMOVE_FORWARD:
				pos_offset = data.direction * offset;
				break;
			case MovementType::kMOVE_BACKWARD:
				pos_offset = -data.direction * offset;
				break;
			case MovementType::kMOVE_LEFT:
				pos_offset = -data.right * offset;
				break;
			case MovementType::kMOVE_RIGHT:
				pos_offset = data.right * offset;
				break;
			default:
				break;
		};
		set_position(data.position + pos_offset);
	}

	void set_direction(const Float3 direction) {
		data.direction = direction;
		data.right = normalize(cross(data.direction, data.up));
		m_is_view_dirty = true;
	}

	void set_direction(const float pitch, const float yaw) {
		Float3 direction;
		direction.x = cos(yaw) * cos(pitch);
		direction.y = sin(yaw) * cos(pitch);
		direction.z = sin(pitch);
		// local to world
		set_direction(direction);
	}

	void set_up(const Float3& up) {
		data.up = up;
		data.right = normalize(cross(data.direction, data.up));
		m_is_view_dirty = true;
	}

	void set_near(Float near) {
		data.near = near;
		m_is_projection_dirty = true;
	}
	void set_far(Float far) {
		data.far = far;
		m_is_projection_dirty = true;
	}
	void set_fov(Float fov) {
		data.fov = fov;
		m_is_projection_dirty = true;
	}
	void set_aspect(Float aspect) {
		data.aspect = aspect;
		m_is_projection_dirty = true;
	}

	const Float3& get_position() const {
		return data.position;
	}
	const Float3& get_direction() const {
		return data.direction;
	}
	const Float3& get_up() const {
		return data.up;
	}
	const Float3& get_right() const {
		return data.right;
	}
	Float get_near() const {
		return data.near;
	}
	Float get_far() const {
		return data.far;
	}
	Float get_fov() const {
		return data.fov;
	}
	Float get_aspect() const {
		return data.aspect;
	}

protected:
	// cached statte, view and projection matrix
	mutable bool m_is_view_dirty = true;
	mutable bool m_is_projection_dirty = true;
	mutable Float4x4 m_view_matrix = {};
	mutable Float4x4 m_projection_matrix = {};
};

}// namespace sail