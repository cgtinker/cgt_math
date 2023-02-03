#[macro_use]
mod macros;

mod algebra;
pub use algebra::f32_extension::ProceduralOperators;
pub use algebra::vec2::Vector2;
pub use algebra::vec3::Vector3;
pub use algebra::vec4::Vector4;
pub use algebra::quat::Quaternion;
pub use algebra::mat3::RotationMatrix;
pub use algebra::mat4::Mat4x4;
pub use algebra::euler::Euler;
pub use algebra::euler::EulerOrder;

mod geometry;
pub use geometry::plane::Plane;
pub use geometry::points::Points;
