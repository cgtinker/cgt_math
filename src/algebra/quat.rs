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
    /// Create new quaternion.
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector4};
    /// let a = Quaternion::IDENTITY;
    /// let b = Quaternion::wxyz(1.0, 0.0, 0.0, 0.0);
    /// assert_eq!(a, b);
    /// ```
    pub const IDENTITY: Self = Self::new(1.0, 0.0, 0.0, 0.0);
    pub const NAN: Self = Self::new(f32::NAN, f32::NAN, f32::NAN, f32::NAN);

    /// Create new quaternion.
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector4};
    /// let a = Quaternion::new(5.0, 2.0, 1.0, 3.0);
    /// let b = Quaternion::wxyz(5.0, 2.0, 1.0, 3.0);
    /// assert_eq!(a, b);
    /// ```
    #[inline]
    pub const fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { v: Vector4 { x, y, z, w } }
    }

    /// Create new quaternion.
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector4};
    /// let a = Quaternion::new(5.0, 2.0, 1.0, 3.0);
    /// let b = Quaternion::xyzw(2.0, 1.0, 3.0, 5.0);
    /// assert_eq!(a, b);
    /// ```
    #[inline]
    pub const fn xyzw(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { v: Vector4 { x, y, z, w } }
    }

    /// Create new quaternion.
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector4};
    /// let a = Quaternion::new(5.0, 2.0, 1.0, 3.0);
    /// let b = Quaternion::wxyz(5.0, 2.0, 1.0, 3.0);
    /// assert_eq!(a, b);
    /// ```
    #[inline]
    pub const fn wxyz(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { v: Vector4 { x, y, z, w } }
    }

    /// Create new quaternion from vector. 
    /// Quaternion Order: [w, x, y, z]
    /// Vector Order: [x, y, z, w]
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector4};
    /// let v = Vector4::new(5.0, 2.0, 3.0, 1.0);
    /// let q = Quaternion::wxyz(1.0, 5.0, 2.0, 3.0);
    /// let vq = Quaternion::from_vec(v);
    /// assert_eq!(q, vq);
    /// ```
    #[inline]
    pub const fn from_vec(v: Vector4) -> Self {
        Self { v }
    }

    /// Create new quaternion from array. [w, x, y, z]
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector4};
    /// let q = Quaternion::from_array([1.0, 2.3, 1.0, 0.0]);
    /// assert_eq!(q, Quaternion::wxyz(1.0, 2.3, 1.0, 0.0))
    /// ```
    #[inline]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self {
            v: Vector4 {
                x: a[1],
                y: a[2],
                z: a[3],
                w: a[0],
            }
        }
    }

    /// Returns quaternion as array: [w, y, x, z]
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector4};
    /// let v = Quaternion::from_array([1.0, 2.3, 1.0, 0.0]);
    /// let arr = v.to_array();
    /// assert_eq!(arr, [1.0, 2.3, 1.0, 0.0]);
    /// ```
    #[inline]
    pub const fn to_array(&self) -> [f32; 4] {
        [self.v.w, self.v.x, self.v.y, self.v.z]
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
    /// let q2 = Quaternion::xyzw(-0.32531965, -0.16741525, -0.82751817, 0.42585564);
    /// assert_eq!(v, q2);
    /// ```
    #[inline]
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

    /// Returns the rotation between to vectors.
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector3};
    /// let v1 = Vector3::new(1.0, 2.0, 0.5);
    /// let v2 = Vector3::new(0.5, 0.5, 2.0);
    /// let q = Quaternion::rotation_between(v1.normalize(), v2.normalize());
    /// assert_eq!(q.fround(4), Quaternion::xyzw(0.4433207, -0.20688298, -0.059109423, 0.87015647).fround(4));
    /// ```
    #[inline]
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
            s*0.5f32,
            rot_axis.x*invs,
            rot_axis.y*invs,
            rot_axis.z*invs,
        )
    }

    /// Returns a quaternion from an angle to an axis.
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector3};
    /// use std::f32::consts::PI;
    /// let v1 = Vector3::new(0.0, 2.0, 1.0);
    /// let q = Quaternion::from_axis_angle(v1, PI/2.0);
    /// assert_eq!(q.fround(4), Quaternion::xyzw(0.0, 1.4142135, 0.70710677, 0.70710677).fround(4));
    /// ```
    #[inline]
    pub fn from_axis_angle(axis: Vector3, angle: f32) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        let v = axis * s;
        Self::new(c, v.x, v.y, v.z)
    }

    /// Returns the rotation to the x-axis from an angle.
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion};
    /// use std::f32::consts::PI;
    /// let q = Quaternion::from_rotation_x(PI/2.0);
    /// assert_eq!(q.fround(4), Quaternion::new(0.70710677, 0.70710677, 0.0, 0.0).fround(4));
    /// ```
    #[inline]
    pub fn from_rotation_x(angle: f32) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        Self::new(c, s, 0.0, 0.0)
    }

    /// Returns the rotation to the y-axis from an angle.
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion};
    /// use std::f32::consts::PI;
    /// let q = Quaternion::from_rotation_y(PI/2.0);
    /// assert_eq!(q.fround(4), Quaternion::new(0.70710677, 0.0, 0.70710677, 0.0).fround(4));
    /// ```
    #[inline]
    pub fn from_rotation_y(angle: f32) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        Self::new(c, 0.0, s, 0.0)
    }

    /// Returns the rotation to the z-axis from an angle.
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion};
    /// use std::f32::consts::PI;
    /// let q = Quaternion::from_rotation_z(PI/2.0);
    /// assert_eq!(q.fround(4), Quaternion::new(0.70710677, 0.0, 0.0, 0.70710677).fround(4));
    /// ```
    #[inline]
    pub fn from_rotation_z(angle: f32) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        Self::new(c, 0.0, 0.0, s)
    }

    /// Converts quaternion to rotation matrix..
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, RotationMatrix};
    /// let q = Quaternion::IDENTITY;
    /// let m = q.quat_to_rotation_matrix();
    /// assert_eq!(m, RotationMatrix::IDENTITY);
    /// ```
    #[inline]
    pub fn quat_to_rotation_matrix(&self) -> RotationMatrix {
        RotationMatrix::from_quaternion(*self)
    }

    /// Create quaternion from the columns of a 3x3 rotation matrix.
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, Vector3};
    /// let x = Vector3::new(1.0, 0.0, 0.0);
    /// let y = Vector3::new(0.0, 1.0, 0.0);
    /// let z = Vector3::new(0.0, 0.0, 1.0);
    /// let q = Quaternion::from_rotation_axes(x, y, z);
    /// assert_eq!(q, Quaternion::IDENTITY);
    /// ```
    #[inline]
    pub fn from_rotation_axes(x: Vector3, y: Vector3, z: Vector3) -> Self {
        // Based on https://github.com/bitshifter/glam-rs && https://github.com/microsoft/DirectXMath `XM$quaternionRotationMatrix`
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
                Self::xyzw(
                    four_xsq * inv4x,
                    (m01 + m10) * inv4x,
                    (m02 + m20) * inv4x,
                    (m12 - m21) * inv4x,
                )
            } else {
                // y^2 >= x^2
                let four_ysq = omm22 + dif10;
                let inv4y = 0.5 / four_ysq.sqrt();
                Self::xyzw(
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
                Self::xyzw(
                    (m02 + m20) * inv4z,
                    (m12 + m21) * inv4z,
                    four_zsq * inv4z,
                    (m01 - m10) * inv4z,
                )
            } else {
                // w^2 >= z^2
                let four_wsq = opm22 + sum10;
                let inv4w = 0.5 / four_wsq.sqrt();
                Self::xyzw(
                    (m12 - m21) * inv4w,
                    (m20 - m02) * inv4w,
                    (m01 - m10) * inv4w,
                    four_wsq * inv4w,
                )
            }
        }
    }

    /// Creates a quaternion from a 3x3 rotation matrix.
    /// # Example
    /// ```
    /// use cgt_math::{Quaternion, RotationMatrix};
    /// let q = Quaternion::from_rotation_matrix(&RotationMatrix::IDENTITY);
    /// assert_eq!(q, Quaternion::IDENTITY);
    /// ```
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
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.v.is_nan()
    }

    /// Returns if any component is infinte.
    /// # Example
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(0.0, 421.0, f32::INFINITY, f32::NAN);
    /// assert!(a.is_infinite());
    /// ```
    #[inline]
    pub fn is_infinite(&self) -> bool {
        self.v.is_infinite()
    }

    /// Returns if all components are finite.
    /// # Example
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(0.0, 421.0, f32::INFINITY, f32::NAN);
    /// let b = Quaternion::new(0.0, 421.0, 1.0, 0.5);
    /// assert!(!a.is_finite());
    /// assert!(b.is_finite());
    /// ```
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
    #[inline]
    pub fn abs(self) -> Self {
        Self { v: self.v.abs() }
    }

    /// Returns vector with rounded values.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(-1.331234, 0.9414214, 2.14245, 1.04124);
    /// let b = Quaternion::new(-1.33, 0.94, 2.14, 1.04);
    /// assert_eq!(a.fround(2), b);
    /// ```
    #[inline]
    pub fn fround(&self, k: u32) -> Self {
        Self { v: self.v.fround(k) }
    }

    /// Returns vector with clamped values.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(-1.3, 0.9, 2.5, 1.0);
    /// let b = Quaternion::new(-1.0, 0.9, 1.0, 1.0);
    /// assert_eq!(a.clamp(-1.0, 1.0), b);
    /// ```
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
    pub fn magnitude(&self) -> f32 {
        self.length()
    }

    /// Returns sum of vector attrs.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(12.0, -3.0, 4.0, 1.0);
    /// assert_eq!(a.sum(), 14.0);
    /// ```
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
    pub fn angle(&self, other: Self) -> f32 {
        self.v.angle(other.v)
    }

    /// Returns normalized quaternion.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(2.0, -4.0, 21.0, 1.0);
    /// let b = Quaternion::new(0.09304842, -0.18609685, 0.97700846, 0.04652421);
    /// assert_eq!(a.normalize(), b);
    /// ```
    #[inline]
    pub fn normalize(&self) -> Self {
        let len = self.v.length();
        if len != 0.0 {
            *self * (1.0/len)
        }
        else {
            Quaternion::IDENTITY
        }
    }

    /// Checks if quaternion is normalized.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(2.0, -4.0, 21.0, 1.0);
    /// assert!(!a.is_normalized());
    /// assert!(a.normalize().is_normalized());
    /// ```
    #[inline]
    pub fn is_normalized(&self) -> bool {
        self.v.is_normalized()
    }

    /// Returns an inverted quaternion.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let q = Quaternion::new(2.0, -4.0, 21.0, 1.0).invert();
    /// assert_eq!(
    ///     q.fround(4),
    ///     Quaternion::xyzw(0.008658, -0.0454, -0.00212, 0.0043).fround(4));
    /// ```
    #[inline]
    pub fn invert(&self) -> Self {
        let f: f32 = self.dot(*self);
        if f == 0.0f32 {
            return *self;
        }
        self.conjugate() * 1.0 / f
    }

    /// Returns the rotation offset from this, to another quaternion.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(2.0, -4.0, 21.0, 1.0);
    /// let b = Quaternion::new(0.5, 1.0, 1.0, 2.0);
    /// assert_eq!(
    ///     a.rotation_offset(b).fround(4), 
    ///     Quaternion::xyzw(-0.08008, -0.03787, 0.061688, 0.0432).fround(4));
    /// ```
    #[inline]
    pub fn rotation_offset(&self, other: Self) -> Self {
        let mut v = self.conjugate();
        v *= 1.0f32 / self.dot(*self);
        v*other
    }

    /// Conjugates Quaternion.
    /// # Example:
    /// ```
    /// use cgt_math::Quaternion;
    /// let a = Quaternion::new(1.0, -3.0, 4.0, 1.0);
    /// let b = Quaternion::new(1.0, 3.0, -4.0, -1.0);
    /// assert_eq!(a.conjugate(), b);
    /// ```
    #[inline]
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
        let rhs = Quaternion::new(-other.v.w, other.v.x, other.v.y, -other.v.z);
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

// Returns w-x-y-z when indexing.
impl Index<usize> for Quaternion {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.v.w,
            1 => &self.v.x,
            2 => &self.v.y,
            3 => &self.v.z,
            _ => panic!("Index Error: {}", index),
        }
    }
}
