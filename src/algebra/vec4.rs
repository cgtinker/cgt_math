use crate::{Vector2, Vector3, F32Utilities};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign, Index};
use std::f32::consts::PI;

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
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Creates a new vector from an array.
    /// # Example
    /// ```
    /// use cgt_math::Vector4;
    /// let vec = Vector4::from_array([1.0, 1.0, 1.0, 1.0]);
    /// assert_eq!(vec, Vector4::ONE);
    /// ```
    #[inline]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }

    /// Creates array from vector
    /// # Example
    /// ```
    /// use cgt_math::Vector4;
    /// let vec = Vector4::from_array([1.0, 1.0, 1.0, 1.0]);
    /// assert_eq!(vec.to_array(), [1.0, 1.0, 1.0, 1.0]);
    /// ```    
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

    /// Returns vector with absolute values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(-1.0, 0.0, 2.0, 1.0);
    /// let b = Vector4::new(1.0, 0.0, 2.0, 1.0);
    /// assert_eq!(a.abs(), b);
    /// ```
    #[inline]
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
    #[inline]
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
    #[inline]
    pub fn floor(&self) -> Self {
        Self {
            x: self.x.floor(),
            y: self.y.floor(),
            z: self.z.floor(),
            w: self.w.floor(),
        }
    }

    /// Returns vector with sinus of vector values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector4, F32Utilities};
    /// use std::f32::consts::PI;
    /// let a = Vector4::new(PI/2.0, PI, -PI/2.0, 0.0);
    /// let b = Vector4::new(1.0, 0.0, -1.0, 0.0);
    /// assert_eq!(a.sin().fround(4), b.fround(4));
    /// ```
    #[inline]
    pub fn sin(&self) -> Self {
        Self {
            x: self.x.sin(),
            y: self.y.sin(),
            z: self.z.sin(),
            w: self.w.sin(),
        }
    }

    /// Returns vector with inverse sinus of vector values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector4, F32Utilities};
    /// use std::f32::consts::PI;
    /// let a = Vector4::new(1.0, 0.0, -1.0, 0.0);
    /// let b = Vector4::new(PI/2.0, 0.0, -PI/2.0, 0.0);
    /// assert_eq!(a.asin().fround(4), b.fround(4));
    /// ```
    #[inline]
    pub fn asin(&self) -> Self {
        Self {
            x: self.x.asin(),
            y: self.y.asin(),
            z: self.z.asin(),
            w: self.w.asin(),
        }
    }

    /// Returns vector with cosinus of vector values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector4, F32Utilities};
    /// use std::f32::consts::PI;
    /// let a = Vector4::new(PI, PI/2.0, 2.0*PI, 0.0);
    /// let b = Vector4::new(-1.0, 0.0, 1.0, 1.0);
    /// assert_eq!(a.cos().fround(4), b.fround(4));
    /// ```
    #[inline]
    pub fn cos(&self) -> Self {
        Self {
            x: self.x.cos(),
            y: self.y.cos(),
            z: self.z.cos(),
            w: self.w.cos(),
        }
    }

    /// Returns vector with inverse cosinus of vector values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector4, F32Utilities};
    /// use std::f32::consts::PI;
    /// let a = Vector4::new(-1.0, 0.0, 1.0, 1.0);
    /// let b = Vector4::new(PI, PI/2.0, 0.0, 0.0);
    /// assert_eq!(a.acos().fround(4), b.fround(4));
    /// ```
    #[inline]
    pub fn acos(&self) -> Self {
        Self {
            x: self.x.acos(),
            y: self.y.acos(),
            z: self.z.acos(),
            w: self.w.acos(),
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
    #[inline]
    pub fn round(&self) -> Self {
        Self {
            x: self.x.round(),
            y: self.y.round(),
            z: self.z.round(),
            w: self.w.round(),
        }
    }

    /// Returns vector with rounded values by k.
    /// # Example
    /// ```
    /// use cgt_math::{F32Utilities, Vector4};
    /// let a = Vector4::new(-1.411, 0.4141, 1.1451, 0.00123);
    /// assert_eq!(a.fround(2), Vector4::new(-1.41, 0.41, 1.14, 0.0));
    /// ```
    #[inline]
    pub fn fround(&self, k: u32) -> Self {
        Self {
            x: self.x.fround(k),
            y: self.y.fround(k),
            z: self.z.fround(k),
            w: self.w.fround(k),
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
    pub fn trunc(&self) -> Self {
        Self {
            x: self.x.trunc(),
            y: self.y.trunc(),
            z: self.z.trunc(),
            w: self.w.trunc(),
        }
    }

    /// Negates vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(12.0, -3.0, 4.0, -1.0);
    /// let b = Vector4::new(-12.0, 3.0, -4.0, 1.0);
    /// assert_eq!(a.neg(), b);
    /// ```
    #[inline]
    pub fn neg(&self) -> Self {
        Self {
            x: self.x.neg(),
            y: self.y.neg(),
            z: self.z.neg(),
            w: self.w.neg(),
        }
    }

    /// Flips a vector at current vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(1.0, 2.0, 4.0, 1.0);
    /// let other = Vector4::new(3.0, 2.0, 4.0, 1.0);
    /// assert_eq!(a.flip(other), Vector4::new(-1.0, 2.0, 4.0, 1.0));
    /// ```
    #[inline]
    pub fn flip(&self, other: Self) -> Self {
        Self {
            x: self.x + (self.x - other.x),
            y: self.y + (self.y - other.y),
            z: self.z + (self.z - other.z),
            w: self.w + (self.w - other.w),
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
    pub fn angle(&self, other: Self) -> f32 {
        (self.dot(other) / (self.length_squared() * other.length_squared()).sqrt()).acos()
    }

    /// Vectors have to be normalized
    /// Returns angle between this and another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(12.0, -3.0, 4.0, 1.0);
    /// let b = Vector4::new(-1.0, 3.0, 4.0, 1.0);
    /// assert_eq!(a.normalize().angle(b.normalize()), 1.6298717);
    /// ```    
    #[inline]
    pub fn angle_normalized(&self, other: Self) -> f32 {
        if self.dot(other) >= 0.0f32 {
            return 2.0 * (other-*self).length() / 2.0;
        }
        PI-2.0 * (other.neg()-*self).length() / 2.0
    }

    /// Returns normalized vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(2.0, -4.0, 21.0, 1.0);
    /// let b = Vector4::new(0.09304842, -0.18609685, 0.97700846, 0.04652421);
    /// assert_eq!(a.normalize(), b);
    /// ``` 
    #[inline]
    pub fn normalize(&self) -> Self {
        *self / self.length()
    }

    /// Returns normalized vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(2.0, -4.0, 21.0, 1.0);
    /// assert!(!a.is_normalized());
    /// assert!(a.normalize().is_normalized())
    /// ``` 
    #[inline]
    pub fn is_normalized(&self) -> bool {
         let len = self.length();
         const MIN: f32 = 1.0f32 - 1.0e-6;
         const MAX: f32 = 1.0f32 + 1.0e-6;

         //len >= MIN && len <= MAX
         (MIN..MAX).contains(&len)
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
    pub fn inverse(&self) -> Self {
        Self::new(1.0 / self.x, 1.0 / self.y, 1.0 / self.z, 1.0 / self.w)
    }

    /// Returns linear interpolation vector. T between [0, 1].
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(1.0, 0.0, 3.0, 1.0);
    /// let b = Vector4::new(2.0, 0.0, 0.0, 0.0);
    /// assert_eq!(a.interpolate(b, 0.5), Vector4::new(1.5, 0.0, 1.5, 0.5));
    /// ```    
    #[inline]
    pub fn interpolate(&self, rhs: Self, t: f32) -> Self {
        const S: f32 = 1.0;
        *self * (S-t) + rhs * t 
    }

    /// Returns cubic interpolation vector. T between [0, 1].
    /// # Example:
    /// ```
    /// use cgt_math::{Vector4, Vector3};
    /// let a = Vector4::new(1.0, 0.0, 3.0, 1.0);
    /// let b = Vector4::new(2.0, 0.0, 0.0, 1.0);
    /// let c = Vector4::new(1.0, 1.0, 1.0, 2.0);
    /// let w = Vector3::new(-1.0, 2.0, 1.0);
    /// assert_eq!(a.interpolate_cubic2(b, c, w), Vector4::new(4.0, 1.0, -2.0, 3.0));
    /// ```
    #[inline]
    pub fn interpolate_cubic2(&self, v1: Self, v2: Self, w: Vector3) -> Self {
        *self * w.x + v1 * w.y + v2 * w.z
    }

    /// Returns cubic interpolation vector. T between [0, 1].
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(1.0, 0.0, 3.0, 1.0);
    /// let b = Vector4::new(2.0, 0.0, 0.0, 1.0);
    /// let c = Vector4::new(1.0, 1.0, 1.0, 2.0);
    /// let d = Vector4::new(1.0, 1.0, 1.0, 2.0);
    /// let w = Vector4::new(-1.0, 2.0, 1.0, 2.0);
    /// assert_eq!(a.interpolate_cubic3(b, c, d, w), Vector4::new(6.0, 3.0, 0.0, 7.0));
    /// ```
    #[inline]
    pub fn interpolate_cubic3(&self, v1: Self, v2: Self, v3: Self, w: Self) -> Self {
        Self {
            x: self.x * w.x + v1.x * w.y + v2.x * w.z + v3.x  * w.w,
            y: self.y * w.x + v1.y * w.y + v2.y * w.z + v3.y  * w.w,
            z: self.z * w.x + v1.z * w.y + v2.z * w.z + v3.z  * w.w,
            w: self.w * w.x + v1.w * w.y + v2.w * w.z + v3.w  * w.w,
        }
    }
    /// Returns center of two points.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(1.0, 0.0, 3.0, 1.0);
    /// let b = Vector4::new(2.0, 0.0, 0.0, 0.0);
    /// assert_eq!(a.center(b), Vector4::new(1.5, 0.0, 1.5, 0.5));
    /// ```
    #[inline]
    pub fn center(&self, other: Self) -> Self {
        Self {
            x: (self.x + other.x)*0.5f32,
            y: (self.y + other.y)*0.5f32,
            z: (self.z + other.z)*0.5f32,
            w: (self.w + other.w)*0.5f32,
        }
    }

    /// Returns center of three points.
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(3.0, 0.0, 3.0, 1.0);
    /// let b = Vector4::new(2.0, 0.0, 0.0, 0.0);
    /// let c = Vector4::new(-2.0, 0.0, 3.0, -1.0);
    /// assert_eq!(a.center_of_three(b, c), Vector4::new(1.0, 0.0, 2.0, 0.0));
    /// ```
    #[inline]
    pub fn center_of_three(&self, v1: Self, v2: Self) -> Self {
        Self {
            x: (self.x + v1.x + v2.x)/3.0f32,
            y: (self.y + v1.y + v2.y)/3.0f32,
            z: (self.z + v1.z + v2.z)/3.0f32,
            w: (self.w + v1.w + v2.w)/3.0f32,
        }
    }

    /// Merges X&Y
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(3.0, 0.0, 3.0, 1.0);
    /// let b = Vector4::new(2.0, 1.0, 0.0, 0.0);
    /// assert_eq!(a.merge_xy(b), Vector4::new(3.0, 2.0, 0.0, 1.0));
    /// ```
    #[inline]
    pub fn merge_xy(&self, rhs: Self) -> Self {
        Self {
            x: self.x,
            y: rhs.x,
            z: self.y,
            w: rhs.y,
        }
    }

    /// Merges Z&W
    /// # Example:
    /// ```
    /// use cgt_math::Vector4;
    /// let a = Vector4::new(-1.0, 0.0, 3.0, 1.0);
    /// let b = Vector4::new(2.0, 1.0, -3.0, 4.0);
    /// assert_eq!(a.merge_xy(b), Vector4::new(-1.0, 2.0, 0.0, 1.0));
    /// ```
    pub fn merge_zw(&self, rhs: Self) -> Self {
        Self {
            x: self.z,
            y: rhs.z,
            z: self.w,
            w: rhs.w,
        }
    }
}


impl Add<f32> for Vector4 {
    type Output = Vector4;
    fn add(self, other: f32) -> Self::Output {
        Vector4 {
            x: self.x + other,
            y: self.y + other,
            z: self.z + other,
            w: self.w + other,
        }
    }
}

impl Add<Vector2> for Vector4 {
    type Output = Vector4;
    fn add(self, other: Vector2) -> Self::Output {
        Vector4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z,
            w: self.w,
        }
    }
}

impl Add<Vector3> for Vector4 {
    type Output = Vector4;
    fn add(self, other: Vector3) -> Self::Output {
        Vector4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w,
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

impl AddAssign<f32> for Vector4 {
    fn add_assign(&mut self, other: f32) {
        self.x += other;
        self.y += other;
        self.z += other;
        self.w += other;
    }
}

impl AddAssign<Vector2> for Vector4 {
    fn add_assign(&mut self, other: Vector2) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl AddAssign<Vector3> for Vector4 {
    fn add_assign(&mut self, other: Vector3) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl AddAssign<Vector4> for Vector4 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl Sub<f32> for Vector4 {
    type Output = Vector4;
    fn sub(self, other: f32) -> Self::Output {
        Vector4 {
            x: self.x - other,
            y: self.y - other,
            z: self.z - other,
            w: self.w - other,
        }
    }
}

impl Sub<Vector2> for Vector4 {
    type Output = Vector4;
    fn sub(self, other: Vector2) -> Self::Output {
        Vector4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z,
            w: self.w,
        }
    }
}

impl Sub<Vector3> for Vector4 {
    type Output = Vector4;
    fn sub(self, other: Vector3) -> Self::Output {
        Vector4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w,
        }
    }
}

impl Sub<Vector4> for Vector4 {
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

impl SubAssign<f32> for Vector4 {
    fn sub_assign(&mut self, other: f32) {
        self.x -= other;
        self.y -= other;
        self.z -= other;
        self.w -= other;
    }
}

impl SubAssign<Vector2> for Vector4 {
    fn sub_assign(&mut self, other: Vector2) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl SubAssign<Vector3> for Vector4 {
    fn sub_assign(&mut self, other: Vector3) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl SubAssign<Vector4> for Vector4 {
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
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

impl Mul<Vector2> for Vector4 {
    type Output = Vector4;
    fn mul(self, other: Vector2) -> Self::Output {
        Vector4 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z,
            w: self.w,
        }
    }
}

impl Mul<Vector3> for Vector4 {
    type Output = Vector4;
    fn mul(self, other: Vector3) -> Self::Output {
        Vector4 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
            w: self.w,
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

impl MulAssign<f32> for Vector4 {
    fn mul_assign(&mut self, other: f32) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
        self.w *= other;    }
}

impl MulAssign<Vector2> for Vector4 {
    fn mul_assign(&mut self, other: Vector2) {
        self.x *= other.x;
        self.y *= other.y;
    }
}

impl MulAssign<Vector3> for Vector4 {
    fn mul_assign(&mut self, other: Vector3) {
        self.x *= other.x;
        self.y *= other.y;
        self.z *= other.z;
    }
}

impl MulAssign<Vector4> for Vector4 {
    fn mul_assign(&mut self, other: Self) {
        self.x *= other.x;
        self.y *= other.y;
        self.z *= other.z;
        self.w *= other.w;
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

impl Div<Vector2> for Vector4 {
    type Output = Vector4;
    fn div(self, other: Vector2) -> Self::Output {
        Vector4 {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z,
            w: self.w,
        }
    }
}

impl Div<Vector3> for Vector4 {
    type Output = Vector4;
    fn div(self, other: Vector3) -> Self::Output {
        Vector4 {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
            w: self.w,
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

impl DivAssign<f32> for Vector4 {
    fn div_assign(&mut self, other: f32) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
        self.w /= other;
       
    }
}

impl DivAssign<Vector2> for Vector4 {
    fn div_assign(&mut self, other: Vector2) {
        self.x /= other.x;
        self.y /= other.y;
    }
}

impl DivAssign<Vector3> for Vector4 {
    fn div_assign(&mut self, other: Vector3) {
        self.x /= other.x;
        self.y /= other.y;
        self.z /= other.z;
    }
}

impl DivAssign for Vector4 {
    fn div_assign(&mut self, other: Self) {
        self.x /= other.x;
        self.y /= other.y;
        self.z /= other.z;
        self.w /= other.w;
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
