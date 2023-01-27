use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::fmt;

use crate::{Vector3, Quaternion};

#[derive(Clone, Copy)]
pub struct RotationMatrix {
    pub x: Vector3,
    pub y: Vector3,
    pub z: Vector3,
}

impl RotationMatrix {
    pub const fn new(x0: f32, x1: f32, x2: f32, y0: f32, y1: f32, y2: f32, z0: f32, z1: f32, z2: f32) -> Self {
        Self {
            x: Vector3::new(x0, x1, x2),
            y: Vector3::new(y0, y1, y2),
            z: Vector3::new(z0, z1, z2),
        }
    }

    pub const fn from_array(mat: [[f32; 3]; 3]) -> Self {
        Self {
            x: Vector3::from_array(mat[0]),
            y: Vector3::from_array(mat[1]),
            z: Vector3::from_array(mat[2]),
        }
    }

    pub const fn to_array(self) -> [[f32; 3]; 3] {
        [
            self.x.to_array(),
            self.y.to_array(),
            self.z.to_array(),
        ]
    }

    pub const fn from_vecs(v1: Vector3, v2: Vector3, v3: Vector3) -> Self {
        Self {
            x: v1,
            y: v2,
            z: v3,
        }
    }

    pub const ZERO: Self = Self::from_vecs(Vector3::ZERO, Vector3::ZERO, Vector3::ZERO);
    pub const IDENTITY: Self = Self::from_vecs(Vector3::X, Vector3::Y, Vector3::Z);

    // Quaternion hast to be normalized
    #[inline]
    pub fn from_quaternion(quat: Quaternion) -> Self {
        let x2 = quat.q.x + quat.q.x;
        let y2 = quat.q.y + quat.q.y;
        let z2 = quat.q.z + quat.q.z;
        let xx = quat.q.x * x2;
        let xy = quat.q.x * y2;
        let xz = quat.q.x * z2;
        let yy = quat.q.y * y2;
        let yz = quat.q.y * z2;
        let zz = quat.q.z * z2;
        let wx = quat.q.w * x2;
        let wy = quat.q.w * y2;
        let wz = quat.q.w * z2;
        Self::from_array([
            [1.0-(yy+zz), xy+wz, wz-wy],
            [xy-wz, 1.0-(xx+zz), yz+wx],
            [xz+wy, yz-wx, 1.0-(xx+yy)],
        ])
    }

    // axis has to be normalized
    #[inline]
    pub fn from_axis_angle(vec: Vector3, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        let xs = vec.x*s;
        let ys = vec.y*s;
        let zs = vec.z*s;

        let r = 1.0 - c;
        let xyr = vec.x*vec.y*r;
        let xzr = vec.x*vec.z*r;
        let yzr = vec.y*vec.z*r;

        Self::from_array(
            [
                [vec.x*vec.x*r+c, xyr+zs, xzr-ys],
                [xyr-zs, vec.y*vec.y*r+c, yzr+xs],
                [xzr+ys, yzr-xs, vec.z*vec.z*r+c],
            ]
        )
    }

    pub fn from_euler() {}

    #[inline]
    pub fn from_rotation_x(angle: f32) -> Self {
        let (sin_a, cos_a) = angle.sin_cos();
        Self::from_vecs(
            Vector3::X,
            Vector3::new(0.0, cos_a, sin_a),
            Vector3::new(0.0, -sin_a, cos_a),
        )
    }

    #[inline]
    pub fn from_rotation_y(angle: f32) -> Self {
        let (sin_a, cos_a) = angle.sin_cos();
        Self::from_vecs(
            Vector3::new(cos_a, 0.0, -sin_a),
            Vector3::Y,
            Vector3::new(sin_a, 0.0, cos_a),
        )
    }


    #[inline]
    pub fn from_rotation_z(angle: f32) -> Self {
        let (sin_a, cos_a) = angle.sin_cos();
        Self::from_vecs(
            Vector3::new(cos_a, sin_a, 0.0),
            Vector3::new(-sin_a, cos_a, 0.0),
            Vector3::Z,
        )
    }

    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }


    /// Returns the transpose of `self`.
    #[must_use]
    #[inline]
    pub fn transpose(&self) -> Self {
        Self {
            x: Vector3::new(self.x.x, self.y.x, self.z.x),
            y: Vector3::new(self.x.y, self.y.y, self.z.y),
            z: Vector3::new(self.x.z, self.y.z, self.z.z),
        }
    }

    /// Returns the determinant of `self`.
    #[inline]
    pub fn determinant(&self) -> f32 {
        self.z.dot(self.x.cross(self.y))
    }
}

impl Default for RotationMatrix {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Add<RotationMatrix> for RotationMatrix {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self::Output {
        Self::from_vecs(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
         )
    }
}

impl AddAssign<RotationMatrix> for RotationMatrix {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = self.add(other);
    }
}

impl Sub<RotationMatrix> for RotationMatrix {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self::Output {
        Self::from_vecs(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
         )
    }
}

impl SubAssign<RotationMatrix> for RotationMatrix {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = self.sub(other);
    }
}

impl Mul<RotationMatrix> for RotationMatrix {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self::Output {
        Self::from_vecs(
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
         )
    }
}

impl MulAssign<RotationMatrix> for RotationMatrix {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = self.mul(other);
    }
}

impl Neg for RotationMatrix {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::from_vecs(
            self.x.neg(),
            self.y.neg(),
            self.z.neg(),
        )
    }
}

impl PartialEq for RotationMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}

impl fmt::Debug for RotationMatrix {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct(stringify!(RotationMatrix))
            .field("x", &self.x)
            .field("y", &self.y)
            .field("z", &self.z)
            .finish()
    }
}
