use super::vec3;
use vec3::Vector3;

#[derive(Clone, Debug, PartialEq)]
pub struct RotationMatrix {
    pub x: Vector3,
    pub y: Vector3,
    pub z: Vector3,
}

impl RotationMatrix {
    pub fn new(r1: [f32; 3], r2: [f32; 3], r3: [f32; 3]) -> Self {
        Self {
            x: Vector3::from_array(r1),
            y: Vector3::from_array(r2),
            z: Vector3::from_array(r3),
        }
    }

    pub fn from_vecs(v1: Vector3, v2: Vector3, v3: Vector3) -> Self {
        Self {
            x: v1,
            y: v2,
            z: v3,
        }
    }
}

