use super::{vec3, vec4, mat3};

use vec3::Vector3;
use vec4::Vector4;
use mat3::RotationMatrix;

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign, Index};

#[derive(Copy, Clone, Debug)]
pub struct Quaternion {
    q: Vector4,
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

    /// Create new quaternion.
    /// # Example
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(3.0, 2.0, 1.0, 5.0);
    /// ```
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { q: Vector4 { x, y, z, w } }
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
        Self { q: vec }
    }

    /// Create new quaternion from array..
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector4};
    /// let q = Quaternion::from_array([1.0, 2.3, 1.0, 0.0]);
    /// ```
    #[inline]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self { q: Vector4::from_array(a) }
    }

    /// `[x, y, z, w]`
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector4};
    /// let q = Quaternion::from_array([1.0, 2.3, 1.0, 0.0]);
    /// let arr = q.to_array();
    /// ```
    #[inline]
    pub const fn to_array(&self) -> [f32; 4] {
        self.q.to_array()
   }

    // TODO:  might have to swite q&&x
    // https://www.euclideanspace.com/maths/algebra/vectors/angleBetween/index.htm
    pub fn from_angle_between_vectors(v1: Vector3, v2: Vector3) -> Self {
        let d = v1.dot(v2);
        let axis = v1.cross(v2);

        let qw = (v1.length_squared()*v2.length_squared()).sqrt() + d;
        if qw < 0.0001 {
            let res = Self { q: Vector4 { x: 0.0, y: -v1.z, z: v1.y, w: v1.x } };
            return res.normalize();
        }
        let res = Self { q: Vector4 { x: qw, y: axis.x, z: axis.y, w: axis.z } };
        return res.normalize();
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

    /// From the columns of a 3x3 rotation matrix.
    #[inline]
    pub(crate) fn from_rotation_axes(x: Vector3, y: Vector3, z: Vector3) -> Self {
        // Based on https://github.com/microsoft/DirectXMath `XM$quaternionRotationMatrix`
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
        self.q.is_nan()
    }

    /// Returns if any vector component is infinte.
    /// # Example
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(0.0, 421.0, f32::INFINITY, f32::NAN);
    /// assert!(a.is_infinite());
    /// ```
    pub fn is_infinite(&self) -> bool {
        self.q.is_infinite()
    }

    /// Returns `true` if, and only if, all elements are finite.  If any element is either
    /// `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.q.is_finite()
    }

    /// Returns if any vector component is infinte.
    /// # Example
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(12.0, 2.0, -1.0, 2.0);
    /// assert_eq!(a.reset(0.0, 0.0, 0.0, 1.0), Quaternion::IDENTITY);
    /// ```
    pub fn reset(&self, x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { q: self.q.reset(x,y,z,w) }
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
        Self { q: self.q.abs() }
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
        Self { q: self.q.ceil() }
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
        Self { q: self.q.floor() }
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
        Self { q: self.q.round() }
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
        Self { q: self.q.clamp(min, max) }
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
        Self { q: self.q.powf(var) }
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
        Self { q: self.q.min(&other.q) }
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
        Self { q: self.q.max(&other.q) }
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
        Self { q: self.q.trunc() }
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
        self.q.dot(other.q)
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
        self.q.sum()
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
        self.q.distance_to_squared(other.q)
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
        self.q.angle(other.q)
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
        Self { q: self.q.normalize() }
    }

    /// Returns the inverse of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// Quaternion::new(42.0, 1.0, 3.0, 1.0).inverse();
    /// Quaternion::new(0.023809524, 1.0, 0.33333334, 1.0);
    /// ```
    pub fn inverse(&self) -> Self {
        Self { q: self.q.inverse() }
    }

    pub fn conjugate(&self) -> Self {
        Self { q: Vector4 { x: -self.q.x, y: -self.q.y, z: -self.q.z, w: self.q.w } }
    }

    /// Performs a linear interpolation between `self` and `rhs` based on the value `s`.
    ///
    /// When `s` is `0.0`, the result will be equal to `self`.  When `s` is `1.0`, the result
    /// will be equal to `rhs`. When `s` is outside of range `[0, 1]`, the result is linearly
    /// extrapolated.
    #[doc(alias = "mix")]
    #[inline]
    pub fn lerp(self, rhs: Self, s: f32) -> Self {
        Self { q: self.q.lerp(rhs.q, s) }
    }
}

impl Add for Quaternion {
    type Output = Quaternion;
    fn add(self, other: Self) -> Self::Output {
        Self { q: Vector4::add(self.q, other.q) }
    }
}

impl AddAssign for Quaternion {
    fn add_assign(&mut self, other: Self) {
        *self = Self { q: self.q + other.q }
    }
}

impl Sub for Quaternion {
    type Output = Quaternion;
    fn sub(self, other: Self) -> Self::Output {
        Self { q: Vector4::sub(self.q, other.q) }
    }
}

impl SubAssign for Quaternion {
    fn sub_assign(&mut self, other: Self) {
        *self = Self { q: self.q - other.q }
    }
}

// Scalar multiplication
impl Mul<f32> for Quaternion {
    type Output = Quaternion;
    fn mul(self, val: f32) -> Self::Output {
        Self { q: Vector4::mul(self.q, val) }
    }
}

impl Mul<Quaternion> for Quaternion {
    type Output = Quaternion;
    fn mul(self, other: Self) -> Self::Output {
        let q1 = self.q;
        let q2 = other.q;

        let x = q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x;
        let y = -q1.x * q2.z + q1.y * q2.w + q1.z * q2.x + q1.w * q2.y;
        let z =  q1.x * q2.y - q1.y * q2.x + q1.z * q2.w + q1.w * q2.z;
        let w = -q1.x * q2.x - q1.y * q2.y - q1.z * q2.z + q1.w * q2.w;
        Self { q: Vector4 { x: x, y: y, z: z, w: w } }
    }
}

impl MulAssign for Quaternion {
    fn mul_assign(&mut self, other: Self) {
        *self = Self { q: self.q * other.q }
    }
}

impl Div<f32> for Quaternion {
    type Output = Quaternion;
    fn div(self, val: f32) -> Self::Output {
        Self { q: Vector4::div(self.q, val) }
    }
}

impl Div<Quaternion> for Quaternion {
    type Output = Quaternion;
    fn div(self, other: Self) -> Self::Output {
        Self { q: Vector4::div(self.q, other.q) }
    }
}

impl DivAssign for Quaternion {
    fn div_assign(&mut self, other: Self) {
        *self = Self { q: self.q / other.q }
    }
}

impl Neg for Quaternion {
    type Output = Quaternion;
    fn neg(self) -> Self::Output {
        Self { q: Vector4::neg(self.q) }
    }
}

impl PartialEq for Quaternion {
    fn eq(&self, other: &Self) -> bool {
        self.q == other.q
    }
}


impl Index<usize> for Quaternion {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.q.x,
            1 => &self.q.y,
            2 => &self.q.z,
            3 => &self.q.w,
            _ => panic!("Index Error: {}", index),
        }
    }
}
