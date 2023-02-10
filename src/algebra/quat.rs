use crate::RotationMatrix;
use crate::Vector3;
use crate::Vector4;
use std::f32::consts::PI;

use std::ops::{Add, Index, Mul, MulAssign, Neg, Sub, Div};

#[derive(Copy, Clone, Debug)]
pub struct Quaternion {
    pub v: Vector4,
}

impl Quaternion {
    /// Create identity quaternion.
    /// # Example
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::IDENTITY;
    /// let b = Quaternion::new(0.0, 0.0, 0.0, 1.0);
    /// assert_eq!(a, b);
    /// ```
    pub const IDENTITY: Self = Self::new(0.0, 0.0, 0.0, 1.0);
    pub const NAN: Self = Self::new(f32::NAN, f32::NAN, f32::NAN, f32::NAN);

    /// Create new quaternion.
    /// # Example
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(3.0, 2.0, 1.0, 5.0);
    /// ```
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self {
            v: Vector4 { x, y, z, w },
        }
    }

    /// Create new quaternion from vector.
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector4};
    /// let v = Vector4::new(1.0, 2.0, 3.0, 1.0);
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 1.0);
    /// let vq = Quaternion::from_vec(v);
    /// assert_eq!(q, vq);
    /// ```
    pub const fn from_vec(vec: Vector4) -> Self {
        Self { v: vec }
    }

    /// Create new quaternion from array..
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector4};
    /// let v = Quaternion::from_array([1.0, 2.3, 1.0, 0.0]);
    /// ```
    #[inline]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self {
            v: Vector4::from_array(a),
        }
    }

    /// `[x, y, z, w]`
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector4};
    /// let v = Quaternion::from_array([1.0, 2.3, 1.0, 0.0]);
    /// let arr = v.to_array();
    /// ```
    #[inline]
    pub const fn to_array(&self) -> [f32; 4] {
        self.v.to_array()
    }

    // based on opengl rotation tutorial added track and up axis
    /// Returns quaternion rotation based on direction vector,
    /// using an tracking and up axis.
    /// Up & Track Axis type of Vector3::X || Vector3::Y || Vector3::Z
    /// Track != Up
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector3};
    /// let eye = Vector3::new(2.5, 1.5, -3.0);
    /// let dest = Vector3::new(0.0, -2.0, 1.0);
    /// let dir = (dest-eye).normalize();
    /// let v = Quaternion::rotate_towards(dir, Vector3::X, Vector3::Z);
    /// let q2 = Quaternion::new(-0.32531965, -0.16741525, -0.82751817, 0.42585564);
    /// assert_eq!(v, q2);
    /// ```
    pub fn rotate_towards(dir: Vector3, track: Vector3, up: Vector3) -> Self {
        cgt_assert!(dir.is_normalized());
        cgt_assert!(track != up);
        if dir.length_squared() < 0.0001 {
            return Quaternion::IDENTITY;
        }
        let right = dir.cross(Vector3::Z);
        let prev_up = right.cross(dir);
        let q1 = Self::rotation_between(track, dir);
        let new_up = up * q1;
        let q2 = Self::rotation_between(new_up.normalize(), prev_up.normalize());
        q2 * q1
    }

    // based on opengl rotation tutorial
    pub fn rotation_between(start: Vector3, dest: Vector3) -> Self {
        cgt_assert!(start.is_normalized());
        cgt_assert!(dest.is_normalized());

        let cos_theta: f32 = start.dot(dest);
        if cos_theta < -1.0f32+0.0001 {
            return Quaternion::from_axis_angle(start.perpendicular().normalize(), PI/2.0);
        }
        let rot_axis = start.cross(dest);
        let s: f32 = ((1.0f32+cos_theta)*2.0f32).sqrt();
        let invs: f32 = 1.0/s;
        Quaternion::new(
            rot_axis.x*invs,
            rot_axis.y*invs,
            rot_axis.z*invs,
            s*0.5f32
        )
    }

    pub fn from_axis_angle(axis: Vector3, angle: f32) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        let v = axis * s;
        Self::new(v.x, v.y, v.z, c)
    }

    /// Creates a quaternion from the `angle` (in radians) around the x axis.
    #[inline]
    pub fn from_rotation_x(angle: f32) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        Self::new(s, 0.0, 0.0, c)
    }

    /// Creates a quaternion from the `angle` (in radians) around the y axis.
    #[inline]
    pub fn from_rotation_y(angle: f32) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        Self::new(0.0, s, 0.0, c)
    }

    /// Creates a quaternion from the `angle` (in radians) around the z axis.
    #[inline]
    pub fn from_rotation_z(angle: f32) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        Self::new(0.0, 0.0, s, c)
    }

    pub fn quat_to_rotation_matrix(&self) -> RotationMatrix {
        RotationMatrix::from_quaternion(*self)
    }

    /// From the columns of a 3x3 rotation matrix.
    /// Based on https://github.com/microsoft/DirectXMath `XM$quaternionRotationMatrix`
    #[inline]
    pub fn from_rotation_axes(x: Vector3, y: Vector3, z: Vector3) -> Self {
        let (m00, m01, m02) = (x.x, x.y, x.z);
        let (m10, m11, m12) = (y.x, y.y, y.z);
        let (m20, m21, m22) = (z.x, z.y, z.z);
        if m22 <= 0.0 {
            // x^2 + y^2 >= z^2 + w^2
            let dif10 = m11 - m00;
            let omm22 = 1.0 - m22;
            if dif10 <= 0.0 {
                // x^2 >= y^2
                let four_xsq = omm22 - dif10;
                let inv4x = 0.5 / four_xsq.sqrt();
                Self::new(
                    four_xsq * inv4x,
                    (m01 + m10) * inv4x,
                    (m02 + m20) * inv4x,
                    (m12 - m21) * inv4x,
                )
            } else {
                // y^2 >= x^2
                let four_ysq = omm22 + dif10;
                let inv4y = 0.5 / four_ysq.sqrt();
                Self::new(
                    (m01 + m10) * inv4y,
                    four_ysq * inv4y,
                    (m12 + m21) * inv4y,
                    (m20 - m02) * inv4y,
                )
            }
        } else {
            // z^2 + w^2 >= x^2 + y^2
            let sum10 = m11 + m00;
            let opm22 = 1.0 + m22;
            if sum10 <= 0.0 {
                // z^2 >= w^2
                let four_zsq = opm22 - sum10;
                let inv4z = 0.5 / four_zsq.sqrt();
                Self::new(
                    (m02 + m20) * inv4z,
                    (m12 + m21) * inv4z,
                    four_zsq * inv4z,
                    (m01 - m10) * inv4z,
                )
            } else {
                // w^2 >= z^2
                let four_wsq = opm22 + sum10;
                let inv4w = 0.5 / four_wsq.sqrt();
                Self::new(
                    (m12 - m21) * inv4w,
                    (m20 - m02) * inv4w,
                    (m01 - m10) * inv4w,
                    four_wsq * inv4w,
                )
            }
        }
    }

    /// Creates a quaternion from a 3x3 rotation matrix.
    #[inline]
    pub fn from_rotation_matrix(mat: &RotationMatrix) -> Self {
        Self::from_rotation_axes(mat.x, mat.y, mat.z)
    }

    /// Returns if any vector component is nan.
    /// # Example
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(0.0, 421.0, f32::NAN, f32::NAN);
    /// assert!(a.is_nan());
    /// ```
    pub fn is_nan(&self) -> bool {
        self.v.is_nan()
    }

    /// Returns if any vector component is infinte.
    /// # Example
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(0.0, 421.0, f32::INFINITY, f32::NAN);
    /// assert!(a.is_infinite());
    /// ```
    pub fn is_infinite(&self) -> bool {
        self.v.is_infinite()
    }

    /// Returns `true` if, and only if, all elements are finite.  If any element is either
    /// `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.v.is_finite()
    }

    /// Returns vector with absolute values.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(-1.0, 0.0, 2.0, 1.0);
    /// let b = Quaternion ::new(1.0, 0.0, 2.0, 1.0);
    /// assert_eq!(a.abs(), b);
    /// ```
    pub fn abs(self) -> Self {
        Self { v: self.v.abs() }
    }

    /// Returns vector with ceiled values.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(-1.3, 0.9, 2.5, 1.0);
    /// let b = Quaternion::new(-1.0, 1.0, 3.0, 1.0);
    /// assert_eq!(a.ceil(), b);
    /// ```
    pub fn ceil(&self) -> Self {
        Self { v: self.v.ceil() }
    }

    /// Returns vector with floored values.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(-1.3, 0.9, 2.5, 1.0);
    /// let b = Quaternion::new(-2.0, 0.0, 2.0, 1.0);
    /// assert_eq!(a.floor(), b);
    /// ```
    pub fn floor(&self) -> Self {
        Self { v: self.v.floor() }
    }

    /// Returns vector with rounded values.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(-1.3, 0.9, 2.5, 1.0);
    /// let b = Quaternion::new(-1.0, 1.0, 3.0, 1.0);
    /// assert_eq!(a.round(), b);
    /// ```
    pub fn round(&self) -> Self {
        Self { v: self.v.round() }
    }

    /// Returns vector with clamped values.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(-1.3, 0.9, 2.5, 1.0);
    /// let b = Quaternion::new(-1.0, 0.9, 1.0, 1.0);
    /// assert_eq!(a.clamp(-1.0, 1.0), b);
    /// ```
    pub fn clamp(&self, min: f32, max: f32) -> Self {
        Self {
            v: self.v.clamp(min, max),
        }
    }

    /// Returns vector with powed values.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(2.0, 1.0, 3.0, 2.0);
    /// let b = Quaternion::new(4.0, 1.0, 9.0, 4.0);
    /// assert_eq!(a.powf(2.0), b);
    /// ```
    pub fn powf(&self, var: f32) -> Self {
        Self {
            v: self.v.powf(var),
        }
    }

    /// Returns vector with powed values.
    /// Int gets converted to f32.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(2.0, 1.0, 3.0, 1.0);
    /// let b = Quaternion::new(4.0, 1.0, 9.0, 1.0);
    /// assert_eq!(a.pow(2), b);
    /// ```
    pub fn pow(&self, var: i32) -> Self {
        self.powf(var as f32)
    }

    /// Returns vector with min values.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(2.0, 1.0, 3.0, 1.0);
    /// let b = Quaternion::new(4.0, -1.0, 9.0, 1.0);
    /// let c = Quaternion::new(2.0, -1.0, 3.0, 1.0);
    /// assert_eq!(a.min(&b), c);
    /// ```
    pub fn min(&self, other: &Self) -> Self {
        Self {
            v: self.v.min(&other.v),
        }
    }

    /// Returns vector with max values.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(2.0, 1.0, 3.0, 1.0);
    /// let b = Quaternion::new(4.0, -1.0, 9.0, 1.0);
    /// let c = Quaternion::new(4.0, 1.0, 9.0, 1.0);
    /// assert_eq!(a.max(&b), c);
    /// ```
    pub fn max(&self, other: &Self) -> Self {
        Self {
            v: self.v.max(&other.v),
        }
    }

    /// Returns vector with turncated values.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(2.2, 1.9, 3.0, 1.0);
    /// let b = Quaternion::new(2.0, 1.0, 3.0, 1.0);
    /// assert_eq!(a.trunc(), b);
    /// ```
    pub fn trunc(&self) -> Self {
        Self { v: self.v.trunc() }
    }

    /// Returns dot product of this with another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(12.0, -3.0, 4.0, 1.0);
    /// let b = Quaternion::new(3.0, 3.0, 3.0, 1.0);
    /// assert_eq!(a.dot(b), 40.0);
    /// ```
    pub fn dot(&self, other: Self) -> f32 {
        self.v.dot(other.v)
    }

    /// Returns length squared of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(12.0, -3.0, 4.0, 1.0);
    /// assert_eq!(a.length_squared(), 170.0);
    /// ```
    pub fn length_squared(&self) -> f32 {
        self.dot(*self)
    }

    /// Returns length of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(12.0, -3.0, 4.0, 1.0);
    /// assert_eq!(a.length(), 13.038404);
    /// ```
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns length of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(12.0, -3.0, 4.0, 1.0);
    /// assert_eq!(a.length(), 13.038404);
    /// ```
    pub fn magnitude(&self) -> f32 {
        self.length()
    }

    /// Returns sum of vector attrs.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(12.0, -3.0, 4.0, 1.0);
    /// assert_eq!(a.sum(), 13.0);
    /// ```
    pub fn sum(&self) -> f32 {
        self.v.sum()
    }

    /// Returns this distance squared to another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(12.0, -3.0, 4.0, 1.0);
    /// let b = Quaternion::new(-1.0, 3.0, 4.0, 1.0);
    /// assert_eq!(a.distance_to_squared(b), 205.0);
    /// ```
    pub fn distance_to_squared(&self, other: Self) -> f32 {
        self.v.distance_to_squared(other.v)
    }

    /// Returns this distance squared to another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(12.0, -3.0, 4.0, 1.0);
    /// let b = Quaternion::new(-1.0, 3.0, 4.0, 1.0);
    /// assert_eq!(a.distance_to(b), 14.3178215);
    /// ```
    pub fn distance_to(&self, other: Self) -> f32 {
        self.distance_to_squared(other).sqrt()
    }

    /// Returns angle between this and another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(12.0, -3.0, 4.0, 1.0);
    /// let b = Quaternion::new(-1.0, 3.0, 4.0, 1.0);
    /// assert_eq!(a.angle(b), 1.6298717);
    /// ```
    pub fn angle(&self, other: Self) -> f32 {
        self.v.angle(other.v)
    }

    /// Returns normalized vector.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(2.0, -4.0, 21.0, 1.0);
    /// let b = Quaternion::new(0.09304842, -0.18609685, 0.97700846, 0.04652421);
    /// assert_eq!(a.normalize(), b);
    /// ```
    pub fn normalize(&self) -> Self {
        let len = self.v.length();
        if len != 0.0 {
            *self * (1.0/len)
        }
        else {
            Quaternion::IDENTITY
        }
    }

    pub fn is_normalized(&self) -> bool {
        self.v.is_normalized()
    }

    /// Returns the inverse of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// Quaternion::new(42.0, 1.0, 3.0, 1.0).inverse();
    /// Quaternion::new(0.023809524, 1.0, 0.33333334, 1.0);
    /// ```
    pub fn inverse(&self) -> Self {
        Self {
            v: self.v.inverse(),
        }
    }

    pub fn invert(&self) -> Self {
        let f: f32 = self.dot(*self);
        if f == 0.0f32 {
            return *self;
        }
        self.conjugate() * 1.0 / f
    }

    pub fn rotation_between_quats(&self, other: Self) -> Self {
        let mut v = self.conjugate();
        v *= 1.0f32 / self.dot(*self);
        v*other
    }

    pub fn conjugate(&self) -> Self {
        Self {
            v: Vector4 {
                x: -self.v.x,
                y: -self.v.y,
                z: -self.v.z,
                w: self.v.w,
            },
        }
    }

}

impl Add for Quaternion {
    type Output = Quaternion;
    fn add(self, other: Self) -> Self::Output {
        self.mul(other)
    }
}

impl Sub for Quaternion {
    type Output = Quaternion;
    fn sub(self, other: Self) -> Self::Output {
        let rhs = Quaternion::new(other.v.x, other.v.y, other.v.z, -other.v.w);
        self * rhs
    }
}

impl Div<f32> for Quaternion {
    type Output = Quaternion;
    fn div(self, val: f32) -> Self::Output {
        Self {
            v: Vector4::div(self.v, val),
        }
    }
}

// Scalar multiplication
impl Mul<f32> for Quaternion {
    type Output = Quaternion;
    fn mul(self, val: f32) -> Self::Output {
        Self {
            v: Vector4::mul(self.v, val),
        }
    }
}

impl MulAssign<f32> for Quaternion {
    fn mul_assign(&mut self, val: f32) {
        self.v *= val
    }
}

impl Mul<Quaternion> for Quaternion {
    type Output = Quaternion;
    fn mul(self, other: Self) -> Self::Output {

        let a = &self.v;
        let b = &other.v;
        let mut v = Vector4::ZERO;
        v.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
        v.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
        v.y = a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z;
        v.z = a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x;
        Quaternion::from_vec(v)
    }
}

// Quat mul
impl MulAssign<Quaternion> for Quaternion {
    fn mul_assign(&mut self, other: Self) {
        let q1 = self.v;
        let q2 = &other.v;
        self.v.x =  q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x;
        self.v.y = -q1.x * q2.z + q1.y * q2.w + q1.z * q2.x + q1.w * q2.y;
        self.v.z =  q1.x * q2.y - q1.y * q2.x + q1.z * q2.w + q1.w * q2.z;
        self.v.w = -q1.x * q2.x - q1.y * q2.y - q1.z * q2.z + q1.w * q2.w;
    }
}

impl Neg for Quaternion {
    type Output = Quaternion;
    fn neg(self) -> Self::Output {
        Self {
            v: Vector4::neg(&self.v),
        }
    }
}

impl PartialEq for Quaternion {
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v
    }
}

impl Index<usize> for Quaternion {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.v.x,
            1 => &self.v.y,
            2 => &self.v.z,
            3 => &self.v.w,
            _ => panic!("Index Error: {}", index),
        }
    }
}
