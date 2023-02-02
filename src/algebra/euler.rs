use crate::{Quaternion, Vector3};
use std::f32::consts::PI;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign, Index, IndexMut};


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
        Self {
            v: Vector3 { x: x, y: y, z: z },
        }
    }

    pub const fn from_vec(v: Vector3) -> Self {
        Self { v: v }
    }

    pub const ZERO: Self = Self { v: Vector3::ZERO };

    // based on cgmath euler conversion https://github.com/rustgd/cgmath
    pub fn from_quat(quat: Quaternion, order: EulerOrder) -> Self {
        let mut x = Self::heading(quat, order);
        let mut y = Self::attitude(quat, order);
        let mut z = Self::bank(quat, order);

        if x == 0.0f32 {
            x = 0.0;
        }
        if y == 0.0f32 {
            y = 0.0;
        }
        if z == 0.0f32 {
            z = 0.0;
        }

        Self::new(x, y, z)
    }

    fn heading(quat: Quaternion, order: EulerOrder) -> f32 {
        let q = quat.q;
        match order {
            EulerOrder::XYZ => (-2.0 * (q.y * q.z - q.w * q.x))
                .atan2(q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z),
            EulerOrder::XZY => {
                (2.0 * (q.y * q.z + q.w * q.x)).atan2(q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z)
            }
            EulerOrder::YXZ => {
                (2.0 * (q.x * q.z + q.w * q.y)).atan2(q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z)
            }
            EulerOrder::YZX => (-2.0 * (q.x * q.z - q.w * q.y))
                .atan2(q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z),
            EulerOrder::ZYX => {
                (2.0 * (q.x * q.y + q.w * q.z)).atan2(q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z)
            }
            EulerOrder::ZXY => (-2.0 * (q.x * q.y - q.w * q.z))
                .atan2(q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z),
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
            EulerOrder::XYZ => (-2.0 * (q.x * q.y - q.w * q.z))
                .atan2(q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z),
            EulerOrder::XZY => {
                (2.0 * (q.x * q.z + q.w * q.y)).atan2(q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z)
            }
            EulerOrder::YXZ => {
                (2.0 * (q.x * q.y + q.w * q.z)).atan2(q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z)
            }
            EulerOrder::YZX => (-2.0 * (q.y * q.z - q.w * q.x))
                .atan2(q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z),
            EulerOrder::ZYX => {
                (2.0 * (q.y * q.z + q.w * q.x)).atan2(q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z)
            }
            EulerOrder::ZXY => (-2.0 * (q.x * q.z - q.w * q.y))
                .atan2(q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z),
        }
    }

    // based on blenders euler to quaternion conversion https://github.com/blender
    pub fn compatible_euler(&mut self, oldrot: &Euler) {
        const PI_THRESH: f32 = 5.1f32;
        const PI_X2: f32 = PI * 2.0f32;
        let eul = self;
        let mut deul = Euler::ZERO;
        /* correct differences of about 360 degrees first */
        for i in 0..3 {
            deul[i] = eul[i] - oldrot[i];
            if deul[i] > PI_THRESH {
                eul[i] -= ((deul[i] / PI_X2) + 0.5f32).floor() * PI_X2;
                deul[i] = eul[i] - oldrot[i];
            } else if deul[i] < -PI_THRESH {
                eul[i] += ((-deul[i] / PI_X2) + 0.5f32).floor() * PI_X2;
                deul[i] = eul[i] - oldrot[i];
            }
        }

        /* is 1 of the axis rotations larger than 180 degrees and the other small? NO ELSE IF!! */
        if (deul[0]).abs() > 3.2f32 && (deul[1]).abs() < 1.6f32 && (deul[2]).abs() < 1.6f32 {
            if deul[0] > 0.0f32 {
                eul[0] -= PI_X2;
            } else {
                eul[0] += PI_X2;
            }
        }
        if (deul[1]).abs() > 3.2f32 && (deul[2]).abs() < 1.6f32 && (deul[0]).abs() < 1.6f32 {
            if deul[1] > 0.0f32 {
                eul[1] -= PI_X2;
            } else {
                eul[1] += PI_X2;
            }
        }
        if (deul[2]).abs() > 3.2f32 && (deul[0]).abs() < 1.6f32 && (deul[1]).abs() < 1.6f32 {
            if deul[2] > 0.0f32 {
                eul[2] -= PI_X2;
            } else {
                eul[2] += PI_X2;
            }
        }
    }
}

impl Add for Euler {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self {
            v: self.v + other.v,
        }
    }
}

impl AddAssign for Euler {
    fn add_assign(&mut self, other: Self) {
        self.v += other.v
    }
}

impl Sub for Euler {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Self {
            v: self.v - other.v,
        }
    }
}

impl SubAssign for Euler {
    fn sub_assign(&mut self, other: Self) {
        self.v -= other.v
    }
}

impl Mul<f32> for Euler {
    type Output = Self;
    fn mul(self, val: f32) -> Self::Output {
        Euler { v: self.v * val }
    }
}

impl Mul<Euler> for Euler {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        Self {
            v: self.v * other.v,
        }
    }
}

impl MulAssign for Euler {
    fn mul_assign(&mut self, other: Self) {
        self.v *= other.v
    }
}

impl Div<f32> for Euler {
    type Output = Euler;
    fn div(self, val: f32) -> Self::Output {
        Euler { v: self.v / val }
    }
}
impl DivAssign for Euler {
    fn div_assign(&mut self, other: Self) {
        self.v /= other.v
    }
}

impl Neg for Euler {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self { v: self.v.neg() }
    }
}

impl PartialEq for Euler {
    fn eq(&self, other: &Euler) -> bool {
        self.v == other.v
    }
}

impl Index<usize> for Euler {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.v.x,
            1 => &self.v.y,
            2 => &self.v.z,
            _ => panic!("Index Error: {}", index),
        }
    }
}

impl IndexMut<usize> for Euler {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.v.x,
            1 => &mut self.v.y,
            2 => &mut self.v.z,
            _ => panic!("Index Error: {}", index),
        }
    }
}
