use crate::RotationMatrix;
use crate::Vector3;
use crate::Vector4;
use std::f32::consts::PI;

use std::ops::{Add, Index, Mul, MulAssign, Neg, Sub, Div};

#[derive(Copy, Clone, Debug)]
pub struct Quaternion {
    pub q: Vector4,
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
            q: Vector4 { x, y, z, w },
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
        Self {
            q: Vector4::from_array(a),
        }
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

        let qw = (v1.length_squared() * v2.length_squared()).sqrt() + d;
        if qw < 0.0001 {
            let res = Self {
                q: Vector4 {
                    x: 0.0,
                    y: -v1.z,
                    z: v1.y,
                    w: v1.x,
                },
            };
            return res.normalize();
        }
        let res = Self {
            q: Vector4 {
                x: qw,
                y: axis.x,
                z: axis.y,
                w: axis.z,
            },
        };
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

    /// Based on https://github.com/blender/blender/ math_rotation
    pub fn quat_to_rotation_matrix(&self) -> RotationMatrix {
        const SQRT2: f64 = 1.41421356237309504880;
        let q0: f64 = SQRT2 * self.q.w as f64;
        let q1: f64 = SQRT2 * self.q.x as f64;
        let q2: f64 = SQRT2 * self.q.y as f64;
        let q3: f64 = SQRT2 * self.q.z as f64;

        let qda: f64 = q0 * q1;
        let qdb: f64 = q0 * q2;
        let qdc: f64 = q0 * q3;
        let qaa: f64 = q1 * q1;
        let qab: f64 = q1 * q2;
        let qac: f64 = q1 * q3;
        let qbb: f64 = q2 * q2;
        let qbc: f64 = q2 * q3;
        let qcc: f64 = q3 * q3;

        let m00: f32 = (1.0 - qbb - qcc) as f32;
        let m01: f32 = (qdc + qab) as f32;
        let m02: f32 = (-qdb + qac) as f32;

        let m10: f32 = (-qdc + qab) as f32;
        let m11: f32 = (1.0 - qaa - qcc) as f32;
        let m12: f32 = (qda + qbc) as f32;

        let m20: f32 = (qdb + qac) as f32;
        let m21: f32 = (-qda + qbc) as f32;
        let m22: f32 = (1.0 - qaa - qbb) as f32;

        RotationMatrix::new(m00, m01, m02, m10, m11, m12, m20, m21, m22)
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

    // Based on https://github.com/dfelinto/blender from_track_quat mathutil
    pub fn from_vec_to_track_quat(vec: Vector3, mut axis: u8, upflag: u8) -> Self {
        const EPS: f32 = 1e-4f32;
        let mut nor: [f32; 3] = [0.0; 3];
        let mut tvec: [f32; 3] = vec.to_array();
        let mut co: f32;

        assert!(axis != upflag);
        assert!(axis <= 5);
        assert!(upflag <= 2);

        let len = vec.length();
        if len == 0.0f32 {
            return Quaternion::IDENTITY;
        }

        // rotate to axis
        if axis > 2 {
            axis -= 3;
        } else {
            tvec[0] *= -1.0;
            tvec[1] *= -1.0;
            tvec[2] *= -1.0;
        }

        // x-axis
        if axis == 0 {
            nor[0] = 0.0;
            nor[1] = -tvec[2];
            nor[2] = tvec[1];

            if (tvec[1].abs() + tvec[2].abs()) < EPS {
                nor[1] = 1.0;
            }
            co = tvec[0];
        }
        // y-axis
        else if axis == 1 {
            nor[0] = tvec[2];
            nor[1] = 0.0;
            nor[2] = -tvec[0];

            if (tvec[0].abs() + tvec[2].abs()) < EPS {
                nor[2] = 1.0;
            }
            co = tvec[1];
        }
        // z-axis
        else {
            nor[0] = -tvec[1];
            nor[1] = tvec[0];
            nor[2] = 0.0;

            if (tvec[0].abs() + tvec[1].abs()) < EPS {
                nor[0] = 1.0;
            }
            co = tvec[2];
        }
        co /= len;

        // saacos
        if co <= -1.0 {
            co = PI;
        } else if co >= 1.0 {
            co = 0.0;
        } else {
            co = co.acos();
        }

        // q from angle
        let q = Quaternion::from_axis_angle(Vector3::from_array(nor).normalize(), co);
        if axis != upflag {
            let angle: f32;
            let mat = q.quat_to_rotation_matrix();
            let fp = mat.z;

            if axis == 0 {
                if upflag == 1 {
                    angle = 0.5 * fp.z.atan2(fp.y);
                } else {
                    angle = -0.5 * fp.y.atan2(fp.z);
                }
            } else if axis == 1 {
                if upflag == 0 {
                    angle = -0.5 * fp.z.atan2(fp.x);
                } else {
                    angle = 0.5 * fp.x.atan2(fp.z);
                }
            } else {
                if upflag == 0 {
                    angle = 0.5 * -fp.y.atan2(-fp.x);
                } else {
                    angle = -0.5 * -fp.x.atan2(-fp.y);
                }
            }

            let si = angle.sin() / len;
            let mut q2 = Quaternion::new(tvec[0] * si, tvec[1] * si, tvec[2] * si, angle.cos());
            q2 *= q;
            return q2
        }
        return q;
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
        Self {
            q: self.q.reset(x, y, z, w),
        }
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
        Self {
            q: self.q.clamp(min, max),
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
            q: self.q.powf(var),
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
            q: self.q.min(&other.q),
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
            q: self.q.max(&other.q),
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
        Self {
            q: self.q.normalize(),
        }
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
            q: self.q.inverse(),
        }
    }

    pub fn invert(&self) -> Self {
        let f: f32 = self.dot(*self);
        if f == 0.0f32 {
            return *self;
        }
        self.conjugate() * 1.0 / f
    }


    pub fn conjugate(&self) -> Self {
        Self {
            q: Vector4 {
                x: -self.q.x,
                y: -self.q.y,
                z: -self.q.z,
                w: self.q.w,
            },
        }
    }

    /// Performs a linear interpolation between `self` and `rhs` based on the value `s`.
    ///
    /// When `s` is `0.0`, the result will be equal to `self`.  When `s` is `1.0`, the result
    /// will be equal to `rhs`. When `s` is outside of range `[0, 1]`, the result is linearly
    /// extrapolated.
    #[doc(alias = "mix")]
    #[inline]
    pub fn lerp(self, rhs: Self, s: f32) -> Self {
        Self {
            q: self.q.lerp(rhs.q, s),
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
        let rhs = Quaternion::new(other.q.x, other.q.y, other.q.z, -other.q.w);
        self * rhs
    }
}

// Scalar multiplication
impl Mul<f32> for Quaternion {
    type Output = Quaternion;
    fn mul(self, val: f32) -> Self::Output {
        Self {
            q: Vector4::mul(self.q, val),
        }
    }
}

impl Div<f32> for Quaternion {
    type Output = Quaternion;
    fn div(self, val: f32) -> Self::Output {
        Self {
            q: Vector4::div(self.q, val),
        }
    }
}

impl Mul<Quaternion> for Quaternion {
    type Output = Quaternion;
    fn mul(self, other: Self) -> Self::Output {
        let a = &self.q;
        let b = &other.q;
        let w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
        let x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
        let y = a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z;
        let z = a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x;
        Self { q: Vector4 { x: x, y: y, z: z, w: w } }
    }
}

// Quat mul
/// Based on https://github.com/blender/blender/ math_rotation
impl MulAssign for Quaternion {
    fn mul_assign(&mut self, other: Self) {
        let a = self.q;
        let b = other.q;
        self.q.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
        self.q.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
        self.q.y = a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z;
        self.q.z = a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x;
        // following seems to be blenders mathutils quat multiplication (full prod)
        // let w = a[1] * b[1] - a[2] * b[2] - a[3] * b[3] - a[0] * b[0];
        // let x = a[1] * b[2] + a[2] * b[1] + a[3] * b[0] - a[0] * b[3];
        // let y = a[1] * b[3] + a[3] * b[1] + a[0] * b[2] - a[2] * b[0];
        // let z = a[1] * b[0] + a[0] * b[1] + a[2] * b[3] - a[3] * b[2];
    }
}

impl Neg for Quaternion {
    type Output = Quaternion;
    fn neg(self) -> Self::Output {
        Self {
            q: Vector4::neg(&self.q),
        }
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
