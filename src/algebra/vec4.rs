use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign, Index};

#[derive(Copy, Clone, Debug)]
pub struct Vector4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vector4 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 1.0, 1.0, 1.0);
    pub const X: Self = Self::new(1.0, 0.0, 0.0, 0.0);
    pub const Y: Self = Self::new(0.0, 1.0, 0.0, 0.0);
    pub const Z: Self = Self::new(0.0, 0.0, 1.0, 0.0);
    pub const W: Self = Self::new(0.0, 0.0, 0.0, 1.0);

    pub const NEG_X: Self = Self::new(-1.0, 0.0, 0.0, 0.0);
    pub const NEG_Y: Self = Self::new(0.0, -1.0, 0.0, 0.0);
    pub const NEG_Z: Self = Self::new(0.0, 0.0, -1.0, 0.0);
    pub const NEG_W: Self = Self::new(0.0, 0.0, 0.0, -1.0);

    pub const INF: Self = Self::new(f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::INFINITY);
    pub const NAN: Self = Self::new(f32::NAN, f32::NAN, f32::NAN, f32::NAN);
    pub const EPSILON: Self = Self::new(f32::EPSILON, f32::EPSILON, f32::EPSILON, f32::EPSILON);

    /// Create vector.
    /// # Example
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(3.0, 2.0, 1.0, 5.0);
    /// ```
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Creates a new vector from an array.
    #[inline]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }

    /// `[x, y, z, w]`
    #[inline]
    pub const fn to_array(&self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Returns if any vector component is nan.
    /// # Example
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(0.0, 421.0, f32::NAN, f32::NAN);
    /// assert!(a.is_nan());
    /// ```
    pub fn is_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }

    /// Returns if any vector component is infinte.
    /// # Example
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::INF;
    /// assert!(a.is_infinite());
    /// ```
    pub fn is_infinite(&self) -> bool {
        self.x.is_infinite() || self.y.is_infinite() || self.z.is_infinite() || self.z.is_infinite()
    }

    /// Returns true if all vector components are finite.
    /// # Example
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(1.0, 1.0, 1.0, f32::INFINITY);
    /// let b = Vector4::new(1.0, 1.0, 1.0, f32::NAN);
    /// let c = Vector4::new(1.0, 1.0, 1.0, 1.0);
    /// assert!(!a.is_finite());
    /// assert!(!b.is_finite());
    /// assert!(c.is_finite());
    /// ```
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
    /// Returns if any vector component is infinte.
    /// # Example
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(12.0, 2.0, -1.0, 2.0);
    /// assert_eq!(a.reset(0.0, 0.0, 0.0, 0.0), Vector4::ZERO);
    /// ```
    pub fn reset(&self, x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Returns vector with absolute values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(-1.0, 0.0, 2.0, 1.0);
    /// let b = Vector4::new(1.0, 0.0, 2.0, 1.0);
    /// assert_eq!(a.abs(), b);
    /// ```
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
            w: self.w.abs(),
        }
    }

    /// Returns vector with ceiled values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(-1.3, 0.9, 2.5, 1.0);
    /// let b = Vector4::new(-1.0, 1.0, 3.0, 1.0);
    /// assert_eq!(a.ceil(), b);
    /// ```
    pub fn ceil(&self) -> Self {
        Self {
            x: self.x.ceil(),
            y: self.y.ceil(),
            z: self.z.ceil(),
            w: self.w.ceil(),
        }
    }

    /// Returns vector with floored values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(-1.3, 0.9, 2.5, 1.0);
    /// let b = Vector4::new(-2.0, 0.0, 2.0, 1.0);
    /// assert_eq!(a.floor(), b);
    /// ```
    pub fn floor(&self) -> Self {
        Self {
            x: self.x.floor(),
            y: self.y.floor(),
            z: self.z.floor(),
            w: self.w.floor(),
        }
    }

    /// Returns vector with rounded values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(-1.3, 0.9, 2.5, 1.0);
    /// let b = Vector4::new(-1.0, 1.0, 3.0, 1.0);
    /// assert_eq!(a.round(), b);
    /// ```
    pub fn round(&self) -> Self {
        Self {
            x: self.x.round(),
            y: self.y.round(),
            z: self.z.round(),
            w: self.w.round(),
        }
    }

    /// Returns vector with clamped values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(-1.3, 0.9, 2.5, 1.0);
    /// let b = Vector4::new(-1.0, 0.9, 1.0, 1.0);
    /// assert_eq!(a.clamp(-1.0, 1.0), b);
    /// ```
    pub fn clamp(&self, min: f32, max: f32) -> Self {
        Self {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
            z: self.z.clamp(min, max),
            w: self.w.clamp(min, max),
        }
    }

    /// Returns vector with powed values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(2.0, 1.0, 3.0, 2.0);
    /// let b = Vector4::new(4.0, 1.0, 9.0, 4.0);
    /// assert_eq!(a.powf(2.0), b);
    /// ```
    pub fn powf(&self, var: f32) -> Self {
        Self {
            x: self.x.powf(var),
            y: self.y.powf(var),
            z: self.z.powf(var),
            w: self.w.powf(var),
        }
    }

    /// Returns vector with powed values.
    /// Int gets converted to f32.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(2.0, 1.0, 3.0, 1.0);
    /// let b = Vector4::new(4.0, 1.0, 9.0, 1.0);
    /// assert_eq!(a.pow(2), b);
    /// ```
    pub fn pow(&self, var: i32) -> Self {
        self.powf(var as f32)
    }

    /// Returns vector with min values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(2.0, 1.0, 3.0, 1.0);
    /// let b = Vector4::new(4.0, -1.0, 9.0, 1.0);
    /// let c = Vector4::new(2.0, -1.0, 3.0, 1.0);
    /// assert_eq!(a.min(&b), c);
    /// ```
    pub fn min(&self, other: &Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
            w: self.w.min(other.w),
        }
    }

    /// Returns vector with max values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(2.0, 1.0, 3.0, 1.0);
    /// let b = Vector4::new(4.0, -1.0, 9.0, 1.0);
    /// let c = Vector4::new(4.0, 1.0, 9.0, 1.0);
    /// assert_eq!(a.max(&b), c);
    /// ```
    pub fn max(&self, other: &Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
            w: self.w.max(other.w),
        }
    }

    /// Returns vector with turncated values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(2.2, 1.9, 3.0, 1.0);
    /// let b = Vector4::new(2.0, 1.0, 3.0, 1.0);
    /// assert_eq!(a.trunc(), b);
    /// ```
    pub fn trunc(&self) -> Self {
        Self {
            x: self.x.trunc(),
            y: self.y.trunc(),
            z: self.z.trunc(),
            w: self.w.trunc(),
        }
    }

    /// Returns dot product of this with another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(12.0, -3.0, 4.0, 1.0);
    /// let b = Vector4::new(3.0, 3.0, 3.0, 1.0);
    /// assert_eq!(a.dot(b), 40.0);
    /// ```
    pub fn dot(&self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Returns length squared of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(12.0, -3.0, 4.0, 1.0);
    /// assert_eq!(a.length_squared(), 170.0);
    /// ```
    pub fn length_squared(&self) -> f32 {
        self.dot(*self)
    }

    /// Returns length of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(12.0, -3.0, 4.0, 1.0);
    /// assert_eq!(a.length(), 13.038404);
    /// ```
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns length of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(12.0, -3.0, 4.0, 1.0);
    /// assert_eq!(a.length(), 13.038404);
    /// ```
    pub fn magnitude(&self) -> f32 {
        self.length()
    }

    /// Returns sum of vector attrs.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(12.0, -3.0, 4.0, 1.0);
    /// assert_eq!(a.sum(), 13.0);
    /// ```
    pub fn sum(&self) -> f32 {
        self.x + self.y + self.z
    }

    /// Returns this distance squared to another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(12.0, -3.0, 4.0, 1.0);
    /// let b = Vector4::new(-1.0, 3.0, 4.0, 1.0);
    /// assert_eq!(a.distance_to_squared(b), 205.0);
    /// ```
    pub fn distance_to_squared(&self, other: Self) -> f32 {
        (*self - other).length_squared()
    }

    /// Returns this distance squared to another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(12.0, -3.0, 4.0, 1.0);
    /// let b = Vector4::new(-1.0, 3.0, 4.0, 1.0);
    /// assert_eq!(a.distance_to(b), 14.3178215);
    /// ```
    pub fn distance_to(&self, other: Self) -> f32 {
        self.distance_to_squared(other).sqrt()
    }

    /// Returns angle between this and another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(12.0, -3.0, 4.0, 1.0);
    /// let b = Vector4::new(-1.0, 3.0, 4.0, 1.0);
    /// assert_eq!(a.angle(b), 1.6298717);
    /// ```
    pub fn angle(&self, other: Self) -> f32 {
        (self.dot(other) / (self.length_squared() * other.length_squared()).sqrt()).acos()
    }

    /// Returns normalized vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(2.0, -4.0, 21.0, 1.0);
    /// let b = Vector4::new(0.09304842, -0.18609685, 0.97700846, 0.04652421);
    /// assert_eq!(a.normalize(), b);
    /// ```
    pub fn normalize(&self) -> Self {
        self.clone() / self.length()
    }

    /// Returns this vector projected on another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(24.0, -5.0, 4.0, 1.0);
    /// let b = Vector4::new(-1.0, 3.0, 4.0, 1.0);
    /// let c = Vector4::new(0.8148148, -2.4444444, -3.2592592, -0.8148148);
    /// assert_eq!(a.project(b), c);
    /// ```
    pub fn project(&self, other: Self) -> Self {
        other * (self.dot(other) / other.length_squared())
    }

    /// Returns vector reflected from a plane defined by given normal.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(-1.0, 0.0, 2.0, 0.0);
    /// let b = Vector4::new(0.0, 0.0, 1.0, 0.0);
    /// assert_eq!(a.reflect(b), Vector4::new(-1.0, 0.0, -2.0, 0.0));
    /// ```
    pub fn reflect(&self, other: Self) -> Self {
        *self - other * (self.dot(other) * 2.0)
    }

    /// Returns this vector slid along plane defined by the given normal.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(2.0, -4.0, 21.0, 0.0);
    /// let b = Vector4::new(-1.0, 0.0, 1.0, 0.0);
    /// let c = Vector4::new(21.0, -4.0, 2.0, 0.0);
    /// assert_eq!(a.slide(b), c);
    /// ```
    pub fn slide(&self, other: Self) -> Self {
        *self - other * self.dot(other)
    }

    /// Returns the inverse of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// Vector4::new(42.0, 1.0, 3.0, 1.0).inverse();
    /// Vector4::new(0.023809524, 1.0, 0.33333334, 1.0);
    /// ```
    pub fn inverse(&self) -> Self {
        Self::new(1.0 / self.x, 1.0 / self.y, 1.0 / self.z, 1.0 / self.w)
    }

    /// Performs a linear interpolation between `self` and `rhs` based on the value `s`.
    ///
    /// When `s` is `0.0`, the result will be equal to `self`.  When `s` is `1.0`, the result
    /// will be equal to `rhs`. When `s` is outside of range `[0, 1]`, the result is linearly
    /// extrapolated.
    #[doc(alias = "mix")]
    #[inline]
    pub fn lerp(self, rhs: Self, s: f32) -> Self {
        self + ((rhs - self) * s)
    }

    pub fn merge_xy(&self, rhs: Self) -> Self {
        Self {
            x: self.x,
            y: rhs.x,
            z: self.y,
            w: rhs.y,
        }
    }

    pub fn merge_zw(&self, rhs: Self) -> Self {
        Self {
            x: self.z,
            y: rhs.z,
            z: self.w,
            w: rhs.w,
        }
    }
}

impl Add for Vector4 {
    type Output = Vector4;
    fn add(self, other: Self) -> Self::Output {
        Vector4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl AddAssign for Vector4 {
    fn add_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl Sub for Vector4 {
    type Output = Vector4;
    fn sub(self, other: Self) -> Self::Output {
        Vector4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl SubAssign for Vector4 {
    fn sub_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl Mul<f32> for Vector4 {
    type Output = Vector4;
    fn mul(self, val: f32) -> Self::Output {
        Vector4 {
            x: self.x * val,
            y: self.y * val,
            z: self.z * val,
            w: self.w * val,
        }
    }
}

impl Mul<Vector4> for Vector4 {
    type Output = Vector4;
    fn mul(self, other: Self) -> Self::Output {
        Vector4 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
            w: self.w * other.w,
        }
    }
}

impl MulAssign for Vector4 {
    fn mul_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
            w: self.w * other.w,
        }
    }
}

impl Div<f32> for Vector4 {
    type Output = Vector4;
    fn div(self, val: f32) -> Self::Output {
        Vector4 {
            x: self.x / val,
            y: self.y / val,
            z: self.z / val,
            w: self.w / val,
        }
    }
}

impl Div<Vector4> for Vector4 {
    type Output = Vector4;
    fn div(self, other: Self) -> Self::Output {
        Vector4 {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
            w: self.w / other.w,
        }
    }
}

impl DivAssign for Vector4 {
    fn div_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
            w: self.w / other.w,
        }
    }
}

impl Neg for Vector4 {
    type Output = Vector4;
    fn neg(self) -> Self::Output {
        Self {
            x: self.x * -1.0,
            y: self.y * -1.0,
            z: self.z * -1.0,
            w: self.w * -1.0,
        }
    }
}

impl PartialEq for Vector4 {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z && self.w == other.w
    }
}

impl Index<usize> for Vector4 {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Index Error: {}", index),
        }
    }
}
