use std::fmt;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{Euler, Quaternion, Vector3, Vector4};

#[derive(Clone, Copy)]
pub struct RotationMatrix {
    pub x: Vector3,
    pub y: Vector3,
    pub z: Vector3,
}

impl RotationMatrix {
    pub const fn new(
        x0: f32,
        x1: f32,
        x2: f32,
        y0: f32,
        y1: f32,
        y2: f32,
        z0: f32,
        z1: f32,
        z2: f32,
    ) -> Self {
        Self {
            x: Vector3::new(x0, x1, x2),
            y: Vector3::new(y0, y1, y2),
            z: Vector3::new(z0, z1, z2),
        }
    }

    pub fn row(&self, index: usize) -> Vector3 {
        match index {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => panic!("Index Error: {}", index),
        }
    }

    pub fn col(&self, index: usize) -> Vector3 {
        match index {
            0 => Vector3::new(self.x.x, self.y.x, self.z.x),
            1 => Vector3::new(self.x.y, self.y.y, self.z.y),
            2 => Vector3::new(self.x.z, self.y.z, self.z.z),
            _ => panic!("Index Error: {}", index),
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
        [self.x.to_array(), self.y.to_array(), self.z.to_array()]
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

    pub fn normalize(&self) -> Self {
        Self {
            x: self.x.normalize(),
            y: self.y.normalize(),
            z: self.z.normalize(),
        }
    }

    pub fn is_normalized(&self) -> bool {
        self.x.is_normalized() && self.y.is_normalized() && self.z.is_normalized()
    }

    // based on
    // http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    #[inline]
    pub fn from_euler(rot: Euler) -> Self {
        rot.to_rotation_matrix()
    }

    // TODO: rpart of routine
    // based on blenders euler to quaternion conversion https://github.com/blender
    pub fn to_euler_x2(&self) -> (Euler, Euler) {
        cgt_assert!(self.is_normalized());
        let mat: &RotationMatrix = self;
        let cy = mat[0][0].hypot(mat[0][1]);
        let mut eul1 = Euler::ZERO;
        let mut eul2 = Euler::ZERO;
        if cy > 16.0f32 * 1.19209290e-7 {
            eul1[0] = mat[1][2].atan2(mat[2][2]);
            eul1[1] = -mat[0][2].atan2(cy);
            eul1[2] = mat[0][1].atan2(mat[0][0]);
            eul2[0] = -mat[1][2].atan2(-mat[2][2]);
            eul2[1] = -mat[0][2].atan2(-cy);
            eul2[2] = -mat[0][1].atan2(-mat[0][0]);
            (eul1, eul2)
        } else {
            eul1[0] = -mat[2][1].atan2(mat[1][1]);
            eul1[1] = -mat[0][2].atan2(cy);
            eul1[2] = 0.0f32;
            (eul1, eul1)
        }
    }
    // TODO: Compare benchmarks
    // based on blenders quaternion to matrix conversion https://github.com/blender
    #[inline]
    pub fn from_quaternion(quat: Quaternion) -> Self {
        cgt_assert!(quat.is_normalized());
        const SQRT2: f32 = 1.4142135623;
        let q0: f32 = SQRT2 * quat[3];
        let q1: f32 = SQRT2 * quat[0];
        let q2: f32 = SQRT2 * quat[1];
        let q3: f32 = SQRT2 * quat[2];

        let qda = q0 * q1;
        let qdb = q0 * q2;
        let qdc = q0 * q3;
        let qaa = q1 * q1;
        let qab = q1 * q2;
        let qac = q1 * q3;
        let qbb = q2 * q2;
        let qbc = q2 * q3;
        let qcc = q3 * q3;

        let mut mat = RotationMatrix::IDENTITY;
        mat[0][0] = 1.0 - qbb - qcc;
        mat[0][1] = qdc + qab;
        mat[0][2] = -qdb + qac;
        mat[1][0] = -qdc + qab;
        mat[1][1] = 1.0 - qaa - qcc;
        mat[1][2] = qda + qbc;
        mat[2][0] = qdb + qac;
        mat[2][1] = -qda + qbc;
        mat[2][2] = 1.0 - qaa - qbb;
        mat
    }

    // https://www.euclideanspace.com/maths/geometry/rotations/conversions/
    #[inline]
    pub fn from_quat(q: Quaternion) -> Self {
        cgt_assert!(q.is_normalized());
        let xx = q.q.x*q.q.x;
        let xy = q.q.x*q.q.y;
        let xz = q.q.x*q.q.z;
        let xw = q.q.x*q.q.w;

        let yy = q.q.y*q.q.y;
        let yz = q.q.y*q.q.z;
        let yw = q.q.y*q.q.w;

        let zz = q.q.z*q.q.z;
        let zw = q.q.z*q.q.w;
        RotationMatrix::new(
            1.0-2.0*(yy-zz),
            2.0*(xy+zw),
            2.0*(xz-yw),
            2.0*(xy-zw),
            1.0-2.0*(xx-zz),
            2.0*(yz+xw),
            2.0*(xz+yw),
            2.0*(yz-xw),
            1.0-2.0*(xx-yy),
        )
    }

    // TODO: find own routine for that
    // based on blenders euler to quaternion conversion https://github.com/blender
    pub fn to_compatible_euler(&self, oldrot: Euler) -> Euler {
        cgt_assert!(self.is_normalized());
        let (mut eul1, mut eul2) = self.to_euler_x2();
        eul1.compatible_euler(&oldrot);
        eul2.compatible_euler(&oldrot);

        let d1 =
            (eul1[0] - oldrot[0]).abs() + (eul1[1] - oldrot[1]).abs() + (eul1[2] - oldrot[2]).abs();
        let d2 =
            (eul2[0] - oldrot[0]).abs() + (eul2[1] - oldrot[1]).abs() + (eul2[2] - oldrot[2]).abs();
        if d1 > d2 {
            eul2
        } else {
            eul1
        }
    }

    fn diagonal(&self) -> Vector3 {
        Vector3 {
            x: self.x.x,
            y: self.y.y,
            z: self.z.z,
        }
    }

    // sum of the diagonal
    fn trace(&self) -> f32 {
        self.diagonal().sum()
    }
    // TODO: Compare benchmarks
    pub fn to_quat(&self) -> Quaternion {
        // https://www.euclideanspace.com/maths/geometry/rotations/conversions/
        // http://www.cs.ucr.edu/~vbz/resources/quatut.pdf
        cgt_assert!(self.is_normalized());
        let mat = self;
        let trace = mat.trace();
        let half: f32 = 0.5f32;

        if trace >= 0.0f32 {
            let s = (1.0f32 + trace).sqrt();
            let w = half * s;
            let s = half / s;
            let x = (mat[1][2] - mat[2][1]) * s;
            let y = (mat[2][0] - mat[0][2]) * s;
            let z = (mat[0][1] - mat[1][0]) * s;
            Quaternion::new(x, y, z, w)
        } else if (mat[0][0] > mat[1][1]) && (mat[0][0] > mat[2][2]) {
            let s = ((mat[0][0] - mat[1][1] - mat[2][2]) + 1.0f32).sqrt();
            let x = half * s;
            let s = half / s;
            let y = (mat[1][0] + mat[0][1]) * s;
            let z = (mat[0][2] + mat[2][0]) * s;
            let w = (mat[1][2] - mat[2][1]) * s;
            Quaternion::new(x, y, z, w)
        } else if mat[1][1] > mat[2][2] {
            let s = ((mat[1][1] - mat[0][0] - mat[2][2]) + 1.0f32).sqrt();
            let y = half * s;
            let s = half / s;
            let z = (mat[2][1] + mat[1][2]) * s;
            let x = (mat[1][0] + mat[0][1]) * s;
            let w = (mat[2][0] - mat[0][2]) * s;
            Quaternion::new(x, y, z, w)
        } else {
            let s = ((mat[2][2] - mat[0][0] - mat[1][1]) + 1.0f32).sqrt();
            let z = half * s;
            let s = half / s;
            let x = (mat[0][2] + mat[2][0]) * s;
            let y = (mat[2][1] + mat[1][2]) * s;
            let w = (mat[0][1] - mat[1][0]) * s;
            Quaternion::new(x, y, z, w)
        }
    }
    // TODO: Compare benchmarks
    // based on blenders matrix to quaternion conversion https://github.com/blender
    #[inline]
    pub fn to_quaternion(&self) -> Quaternion {
        cgt_assert!(self.is_normalized());
        // check trace of matrix - bad precision if close to -1
        let mat: &RotationMatrix = self;
        let trace = mat[0][0] + mat[1][1] + mat[2][2];

        let mut q = Vector4::ZERO;
        if trace > 0.0f32 {
            let mut s: f32 = 2.0f32 * (1.0f32 + trace).sqrt();
            q.w = 0.25f32 * s;
            s = 1.0f32 / s;
            q.x = (mat[1][2] - mat[2][1]) * s;
            q.y = (mat[2][0] - mat[0][2]) * s;
            q.z = (mat[0][1] - mat[1][0]) * s;
        } else {
            /* Find the biggest diagonal element to choose the best formula.
             * Here trace should also be always >= 0, avoiding bad precision. */
            if mat[0][0] > mat[1][1] && mat[0][0] > mat[2][2] {
                let mut s: f32 = 2.0f32 * (1.0f32 + mat[0][0] - mat[1][1] - mat[2][2]).sqrt();
                q.x = 0.25f32 * s;
                s = 1.0f32 / s;
                q.w = (mat[1][2] - mat[2][1]) * s;
                q.y = (mat[1][0] + mat[0][1]) * s;
                q.z = (mat[2][0] + mat[0][2]) * s;
            } else if mat[1][1] > mat[2][2] {
                let mut s: f32 = 2.0f32 * (1.0f32 + mat[1][1] - mat[0][0] - mat[2][2]).sqrt();
                q.y = 0.25f32 * s;
                s = 1.0f32 / s;

                q.w = (mat[2][0] - mat[0][2]) * s;
                q.x = (mat[1][0] + mat[0][1]) * s;
                q.z = (mat[2][1] + mat[1][2]) * s;
            } else {
                let mut s = 2.0f32 * (1.0f32 + mat[2][2] - mat[0][0] - mat[1][1]).sqrt();
                q.z = 0.25f32 * s;
                s = 1.0f32 / s;

                q.w = (mat[0][1] - mat[1][0]) * s;
                q.x = (mat[2][0] + mat[0][2]) * s;
                q.y = (mat[2][1] + mat[1][2]) * s;
            }

            /* Make sure W is non-negative for a canonical result. */
            if q.w < 0.0f32 {
                q *= -1.0f32;
            }
        }
        let res = Quaternion::from_vec(q);
        res.normalize()
    }

    #[inline]
    pub fn from_axis_angle(vec: Vector3, angle: f32) -> Self {
        cgt_assert!(vec.is_normalized());
        let (s, c) = angle.sin_cos();
        let xs = vec.x * s;
        let ys = vec.y * s;
        let zs = vec.z * s;

        let r = 1.0 - c;
        let xyr = vec.x * vec.y * r;
        let xzr = vec.x * vec.z * r;
        let yzr = vec.y * vec.z * r;

        Self::from_array([
            [vec.x * vec.x * r + c, xyr + zs, xzr - ys],
            [xyr - zs, vec.y * vec.y * r + c, yzr + xs],
            [xzr + ys, yzr - xs, vec.z * vec.z * r + c],
        ])
    }

    #[inline]
    pub fn from_rotation_x(angle: f32) -> Self {
        let (sin_a, cos_a) = angle.sin_cos();
        Self::from_vecs(
            Vector3::X,
            Vector3::new(0.0, cos_a, sin_a),
            Vector3::new(0.0, -sin_a, cos_a),
        )
    }

    pub fn round(&self) -> Self {
        Self::from_vecs(self.x.round(), self.y.round(), self.z.round())
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
        Self::from_vecs(self.x + other.x, self.y + other.y, self.z + other.z)
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
        Self::from_vecs(self.x - other.x, self.y - other.y, self.z - other.z)
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
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            rhs.row(0).dot(self.col(0)),
            rhs.row(0).dot(self.col(1)),
            rhs.row(0).dot(self.col(2)),

            rhs.row(1).dot(self.col(0)),
            rhs.row(1).dot(self.col(1)),
            rhs.row(1).dot(self.col(2)),

            rhs.row(2).dot(self.col(0)),
            rhs.row(2).dot(self.col(1)),
            rhs.row(2).dot(self.col(2)),
        )
    }
}

impl MulAssign<RotationMatrix> for RotationMatrix {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul(rhs)
    }
}

impl Neg for RotationMatrix {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::from_vecs(self.x.neg(), self.y.neg(), self.z.neg())
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

impl Index<usize> for RotationMatrix {
    type Output = Vector3;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index Error: {}", index),
        }
    }
}

impl IndexMut<usize> for RotationMatrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index Error: {}", index),
        }
    }
}
