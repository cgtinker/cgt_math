mod algebra;
pub use algebra::vec3::Vector3;
pub use algebra::vec4::Vector4;
pub use algebra::quat::Quaternion;
pub use algebra::mat3::RotationMatrix;
pub use algebra::mat4::Mat4x4;
pub use algebra::euler::Euler;
pub use algebra::euler::EulerOrder;

mod geometry;
pub use geometry::face::Face;
pub use geometry::circle::Circle;
