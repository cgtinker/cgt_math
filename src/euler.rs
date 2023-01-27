
use crate::{Vector3, Quaternion};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EulerOrder {
    XYZ,
    XZY,
    YZX,
    YXZ,
    ZXY,
    ZYX,
}

impl Default for EulerOrder {
    fn default() -> Self {
        Self::XYZ
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Euler {
    pub v: Vector3,
}

impl Euler {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { v: Vector3 { x: x, y: y, z: z } }
    }

    pub const fn from_vec(v: Vector3) -> Self {
        Self { v: v }
    }

    pub fn from_quat(quat: Quaternion, order: EulerOrder) -> Self {
        let mut x = Self::heading(quat, order);
        let mut y = Self::attitude(quat, order);
        let mut z = Self::bank(quat, order);

        if x == 0.0f32 { x = 0.0; }
        if y == 0.0f32 { y = 0.0; }
        if z == 0.0f32 { z = 0.0; }

        Self::new(x, y, z)
    }

    fn heading(quat: Quaternion, order: EulerOrder) -> f32 {
        let q = quat.q;
        match order {
            EulerOrder::XYZ => (-2.0 * (q.y * q.z - q.w * q.x)).atan2(q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z),
            EulerOrder::XZY => (2.0 * (q.y * q.z + q.w * q.x)).atan2(q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z),
            EulerOrder::YXZ => (2.0 * (q.x * q.z + q.w * q.y)).atan2(q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z),
            EulerOrder::YZX => (-2.0 * (q.x * q.z - q.w * q.y)).atan2(q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z),
            EulerOrder::ZYX => (2.0 * (q.x * q.y + q.w * q.z)).atan2(q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z),
            EulerOrder::ZXY => (-2.0 * (q.x * q.y - q.w * q.z)).atan2(q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z),
        }
    }

    fn attitude(quat: Quaternion, order: EulerOrder) -> f32 {
        let q = quat.q;
        match order {
            EulerOrder::XYZ => (2.0 * (q.x * q.z + q.w * q.y)).clamp(-1.0, 1.0).asin(),
            EulerOrder::XZY => (-2.0 * (q.x * q.y - q.w * q.z)).clamp(-1.0, 1.0).asin(),
            EulerOrder::YXZ => (-2.0 * (q.y * q.z - q.w * q.x)).clamp(-1.0, 1.0).asin(),
            EulerOrder::YZX => (2.0 * (q.x * q.y + q.w * q.z)).clamp(-1.0, 1.0).asin(),
            EulerOrder::ZYX => (-2.0 * (q.x * q.z - q.w * q.y)).clamp(-1.0, 1.0).asin(),
            EulerOrder::ZXY => (2.0 * (q.y * q.z + q.w * q.x)).clamp(-1.0, 1.0).asin(),
        }
    }

    fn bank(quat: Quaternion, order: EulerOrder) -> f32 {
        let q = quat.q;
        match order {
            EulerOrder::XYZ => (-2.0 * (q.x * q.y - q.w * q.z)).atan2(q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z),
            EulerOrder::XZY => (2.0 * (q.x * q.z + q.w * q.y)).atan2(q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z),
            EulerOrder::YXZ => (2.0 * (q.x * q.y + q.w * q.z)).atan2(q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z),
            EulerOrder::YZX => (-2.0 * (q.y * q.z - q.w * q.x)).atan2(q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z),
            EulerOrder::ZYX => (2.0 * (q.y * q.z + q.w * q.x)).atan2(q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z),
            EulerOrder::ZXY => (-2.0 * (q.x * q.z - q.w * q.y)).atan2(q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z),
        }
    }
}

impl PartialEq for Euler {
    fn eq(&self, other: &Euler) -> bool {
        self.v == other.v
    }
}
