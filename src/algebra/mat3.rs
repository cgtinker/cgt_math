use std::fmt;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use std::f32::consts::PI;

use crate::{Euler, Quaternion, Vector3};

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


    pub fn round(&self) -> Self {
        Self::from_vecs(self.x.round(), self.y.round(), self.z.round())
    }

    // based on
    // http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    #[inline]
    pub fn from_euler(rot: Euler) -> Self {
        rot.to_rotation_matrix()
    }

    // TODO: rpart of routine
    // based on blenders euler to quaternion conversion https://github.com/blender
    // reread this
    pub fn to_euler_x2(&self) -> (Euler, Euler) {
        cgt_assert!(self.is_normalized());
        let mat: &RotationMatrix = self;
        // cy vs sy thats the same. like exactly same besides idx swap
        // let sy = (mat[0][0]*mat[0][0] + mat[0][1]*mat[0][1]).sqrt();
        let cy = mat[0][0].hypot(mat[0][1]);
        let mut eul1 = Euler::ZERO;
        let mut eul2 = Euler::ZERO;
        //then
        // if sy < 1e-6 {}...
        if cy > 1.0e-6 {
            // matlab: matrix indecies are swapped (?) [2, 1], [2, 0]....
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


    pub fn to_euler(&self) -> (Euler, Euler) {
        /*
         * http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
         * Input matrix
         * [            cos(y)cos(z)                    cos(y)sin(z)            -sin(y)     ]
         * [ sin(x)sin(y)cos(z)-cos(x)sin(z) (sin(x)sin(y)sin(z)+cos(x)cos(z)  sin(x)cos(y) ]
         * [ cos(x)sin(y)cos(z)+sin(x)sin(z) cos(x)sin(y)sin(z)-sin(x)cos(z)   cos(x)cos(y) ]
         *
         * [ cos(y)cos(z) sin(x)sin(y)cos(z)-cos(x)sin(z)  cos(x)sin(y)cos(z)+sin(x)sin(z) ]
         * [ cos(y)sin(z) (sin(x)sin(y)sin(z)+cos(x)cos(z) cos(x)sin(y)sin(z)-sin(x)cos(z) ]
         * [    -sin(y)             sin(x)cos(y)                     cos(x)cos(y)          ]
         */
        const MIN: f32 = -1.0 + 0.0001;
        const MAX: f32 = 1.0 - 0.0001;
        cgt_assert!(self.is_normalized());
        let m = self;
        if m[2][0] > MIN && m[2][0] < MAX {
            let y = -m[2][0].asin();
            let yy = PI - y;

            let x = m[2][1].atan2(m[2][2]);
            let xx = -m[2][1].atan2(-m[2][2]);

            let z = m[1][0].atan2(m[0][0]);
            let zz = -m[1][0].atan2(-m[0][0]);

            (Euler::new(x, y, z), Euler::new(xx, yy, zz))
        }
        else {
            // gibal lock state
            let z = 0.0f32;
            if m[2][0] < MIN {
                let y = PI/2.0;
                let x = m[0][1].atan2(m[0][2]);
                (Euler::new(x, y, z), Euler::new(x, y, z))
            }
            else {
                let yy = -PI/2.0;
                let xx = -m[0][1].atan2(-m[0][2]);
                (Euler::new(xx, yy, z), Euler::new(xx, yy, z))
            }

        }
    }
    pub fn to_eul(&self) -> (Euler, Euler) {
        /*
         * Default:
         * [            cos(y)cos(z)                    cos(y)sin(z)            -sin(y)     ]
         * [ sin(x)sin(y)cos(z)-cos(x)sin(z) (sin(x)sin(y)sin(z)+cos(x)cos(z)  sin(x)cos(y) ]
         * [ cos(x)sin(y)cos(z)+sin(x)sin(z) cos(x)sin(y)sin(z)-sin(x)cos(z)   cos(x)cos(y) ]
         *
         * http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
         * [ cos(y)cos(z) sin(x)sin(y)cos(z)-cos(x)sin(z)  cos(x)sin(y)cos(z)+sin(x)sin(z) ]
         * [ cos(y)sin(z) (sin(x)sin(y)sin(z)+cos(x)cos(z) cos(x)sin(y)sin(z)-sin(x)cos(z) ]
         * [    -sin(y)             sin(x)cos(y)                     cos(x)cos(y)          ]
         */

        cgt_assert!(self.is_normalized());
        let m = self;
        let si = m[0][0].hypot(m[0][1]);
        if si > 1.0e-6 {
            // y != 180 deg
            let x = m[1][2].atan2(m[2][2]);
            let y = -m[0][2].asin();
            let z = m[0][1].atan2(m[0][0]);

            let yy = PI - y;
            let xx = -m[1][2].atan2(-m[2][2]);
            let zz = -m[0][1].atan2(-m[0][0]);

            (Euler::new(x, y, z), Euler::new(xx, yy, zz))
        }
        else {
            // gibal lock state
            let z = 0.0f32;
            let y = PI/2.0;
            let x = -m[2][1].atan2(m[1][1]);

            let yy = -PI/2.0;
            let xx = m[2][1].atan2(m[1][1]);
            let zz = -m[0][1].atan2(-m[0][0]);

            (Euler::new(x, y, z), Euler::new(xx, yy, zz))
        }
    }


    // https://www.euclideanspace.com/maths/geometry/rotations/conversions/
    #[inline]
    pub fn from_quaternion(q: Quaternion) -> Self {
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
            1.0-2.0*(yy+zz),
            2.0*(xy+zw),
            2.0*(xz-yw),

            2.0*(xy-zw),
            1.0-2.0*(xx+zz),
            2.0*(yz+xw),
            2.0*(xz+yw),

            2.0*(yz-xw),
            1.0-2.0*(xx+yy),
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

    /// Diagonal of the matrix.
    fn diagonal(&self) -> Vector3 {
        Vector3 {
            x: self.x.x,
            y: self.y.y,
            z: self.z.z,
        }
    }

    /// Sum of the diagonal.
    fn trace(&self) -> f32 {
        self.diagonal().sum()
    }

    pub fn to_quaternion(&self) -> Quaternion {
        // https://www.euclideanspace.com/maths/geometry/rotations/conversions/
        // http://www.cs.ucr.edu/~vbz/resources/quatut.pdf
        cgt_assert!(self.is_normalized());
        let mat: &RotationMatrix = self;
        let trace = mat.trace();
        const HALF: f32 = 0.5f32;

        if trace >= 0.0f32 {
            let s = (1.0f32 + trace).sqrt();
            let w = HALF * s;
            let s = HALF / s;
            let x = (mat[1][2] - mat[2][1]) * s;
            let y = (mat[2][0] - mat[0][2]) * s;
            let z = (mat[0][1] - mat[1][0]) * s;
            Quaternion::new(x, y, z, w)
        } else if (mat[0][0] > mat[1][1]) && (mat[0][0] > mat[2][2]) {
            let s = ((mat[0][0] - mat[1][1] - mat[2][2]) + 1.0f32).sqrt();
            let x = HALF * s;
            let s = HALF / s;
            let y = (mat[1][0] + mat[0][1]) * s;
            let z = (mat[0][2] + mat[2][0]) * s;
            let w = (mat[1][2] - mat[2][1]) * s;
            Quaternion::new(x, y, z, w)
        } else if mat[1][1] > mat[2][2] {
            let s = ((mat[1][1] - mat[0][0] - mat[2][2]) + 1.0f32).sqrt();
            let y = HALF * s;
            let s = HALF / s;
            let z = (mat[2][1] + mat[1][2]) * s;
            let x = (mat[1][0] + mat[0][1]) * s;
            let w = (mat[2][0] - mat[0][2]) * s;
            Quaternion::new(x, y, z, w)
        } else {
            let s = ((mat[2][2] - mat[0][0] - mat[1][1]) + 1.0f32).sqrt();
            let z = HALF * s;
            let s = HALF / s;
            let x = (mat[0][2] + mat[2][0]) * s;
            let y = (mat[2][1] + mat[1][2]) * s;
            let w = (mat[0][1] - mat[1][0]) * s;
            Quaternion::new(x, y, z, w)
        }
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

    /// Returns if all elements are finite.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Returns if any elements are NaN.
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    /// Returns the transpose.
    #[must_use]
    #[inline]
    pub fn transpose(&self) -> Self {
        Self {
            x: Vector3::new(self.x.x, self.y.x, self.z.x),
            y: Vector3::new(self.x.y, self.y.y, self.z.y),
            z: Vector3::new(self.x.z, self.y.z, self.z.z),
        }
    }

    /// Returns the determinant.
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
