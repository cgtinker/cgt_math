use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign, Index, IndexMut};
use std::f32::consts::PI;
use crate::{Quaternion, F32Utilities};

#[derive(Copy, Clone, Debug)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}


impl Vector3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 1.0, 1.0);

    pub const X: Self = Self::new(1.0, 0.0, 0.0);
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);

    pub const NEG_X: Self = Self::new(-1.0, 0.0, 0.0);
    pub const NEG_Y: Self = Self::new(0.0, -1.0, 0.0);
    pub const NEG_Z: Self = Self::new(0.0, 0.0, -1.0);

    pub const INF: Self = Self::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    pub const NAN: Self = Self::new(f32::NAN, f32::NAN, f32::NAN);
    pub const EPSILON: Self = Self::new(f32::EPSILON, f32::EPSILON, f32::EPSILON);

    /// Create vector.
    /// # Example
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(0.0, 0.0, 0.0);
    /// assert_eq!(a, Vector3::ZERO);
    /// ```
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Creates a new vector from an array.
    /// # Example
    /// ```
    /// use cgt_math::Vector3;
    /// let vec = Vector3::from_array([1.0, 1.0, 1.0]);
    /// assert_eq!(vec, Vector3::ONE);
    /// ```
    #[inline]
    pub const fn from_array(a: [f32; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }

    /// Creates array from vector
    /// # Example
    /// ```
    /// use cgt_math::Vector3;
    /// let vec = Vector3::from_array([1.0, 1.0, 1.0]);
    /// assert_eq!(vec.to_array(), [1.0, 1.0, 1.0]);
    /// ```
    #[inline]
    pub const fn to_array(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Returns if any vector component is nan.
    /// # Example
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(0.0, 421.0, f32::NAN);
    /// assert!(a.is_nan());
    /// ```
    pub fn is_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    /// Returns if any vector component is infinte.
    /// # Example
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::INF;
    /// assert!(a.is_infinite());
    /// ```
    pub fn is_infinite(&self) -> bool {
        self.x.is_infinite() || self.y.is_infinite() || self.z.is_infinite()
    }

    /// Returns if any vector componet is not finite.
    /// # Example
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(1.0, 1.0, f32::INFINITY);
    /// let b = Vector3::new(1.0, 1.0, f32::NAN);
    /// let c = Vector3::new(1.0, 1.0, 1.0);
    /// assert!(!a.is_finite());
    /// assert!(!b.is_finite());
    /// assert!(c.is_finite());
    /// ```
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Returns vector with absolute values.
    /// # Example
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(-1.2, 1.0, -4.0);
    /// assert_eq!(a.abs(), Vector3::new(1.2, 1.0, 4.0));
    /// ```
    #[inline]
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    /// Returns vector with ceiled values.
    /// # Example
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(-1.2, 1.4, -4.0);
    /// assert_eq!(a.ceil(), Vector3::new(-1.0, 2.0, -4.0));
    /// ```
    #[inline]
    pub fn ceil(&self) -> Self {
        Self {
            x: self.x.ceil(),
            y: self.y.ceil(),
            z: self.z.ceil(),
        }
    }

    /// Returns vector with floored values.
    /// # Example
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(-1.2, 1.4, -4.0);
    /// assert_eq!(a.floor(), Vector3::new(-2.0, 1.0, -4.0));
    /// ```
    #[inline]
    pub fn floor(&self) -> Self {
        Self {
            x: self.x.floor(),
            y: self.y.floor(),
            z: self.z.floor(),
        }
    }

    /// Returns vector with sinus of vector values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector3, F32Utilities};
    /// use std::f32::consts::PI;
    /// let a = Vector3::new(PI/2.0, PI, -PI/2.0);
    /// let b = Vector3::new(1.0, 0.0, -1.0);
    /// assert_eq!(a.sin().fround(4), b.fround(4));
    /// ```
    #[inline]
    pub fn sin(&self) -> Self {
        Self {
            x: self.x.sin(),
            y: self.y.sin(),
            z: self.z.sin(),
        }
    }

    /// Returns vector with inverse sinus of vector values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector3, F32Utilities};
    /// use std::f32::consts::PI;
    /// let a = Vector3::new(1.0, 0.0, -1.0);
    /// let b = Vector3::new(PI/2.0, 0.0, -PI/2.0);
    /// assert_eq!(a.asin().fround(4), b.fround(4));
    /// ```
    #[inline]
    pub fn asin(&self) -> Self {
        Self {
            x: self.x.asin(),
            y: self.y.asin(),
            z: self.z.asin(),
        }
    }

    /// Returns vector with cosinus of vector values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector3, F32Utilities};
    /// use std::f32::consts::PI;
    /// let a = Vector3::new(PI, PI/2.0, 2.0*PI);
    /// let b = Vector3::new(-1.0, 0.0, 1.0);
    /// assert_eq!(a.cos().fround(4), b.fround(4));
    /// ```
    #[inline]
    pub fn cos(&self) -> Self {
        Self {
            x: self.x.cos(),
            y: self.y.cos(),
            z: self.z.cos(),
        }
    }

    /// Returns vector with inverse cosinus of vector values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector3, F32Utilities};
    /// use std::f32::consts::PI;
    /// let a = Vector3::new(-1.0, 0.0, 1.0);
    /// let b = Vector3::new(PI, PI/2.0, 0.0);
    /// assert_eq!(a.acos().fround(4), b.fround(4));
    /// ```
    #[inline]
    pub fn acos(&self) -> Self {
        Self {
            x: self.x.acos(),
            y: self.y.acos(),
            z: self.z.acos(),
        }
    }

    /// Returns vector with rounded values.
    /// # Example
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(-1.411, 0.4141, 1.1451);
    /// assert_eq!(a.round(), Vector3::new(-1.0, 0.0, 1.0));
    /// ```
    #[inline]
    pub fn round(&self) -> Self {
        Self {
            x: self.x.round(),
            y: self.y.round(),
            z: self.z.round(),
        }
    }

    /// Returns vector with rounded values by k.
    /// # Example
    /// ```
    /// use cgt_math::{F32Utilities, Vector3};
    /// let a = Vector3::new(-1.411, 0.4141, 1.1451);
    /// assert_eq!(a.fround(2), Vector3::new(-1.41, 0.41, 1.14));
    /// ```
    pub fn fround(&self, k: u32) -> Self {
        Self {
            x: self.x.fround(k),
            y: self.y.fround(k),
            z: self.z.fround(k),
        }
    }

    /// Returns vector with clamped values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector3};
    /// let a = Vector3::new(-1.411, 0.4141, 1.1451);
    /// assert_eq!(a.clamp(-1.0, 1.0), Vector3::new(-1.0, 0.4141, 1.0));
    /// ```
    #[inline]
    pub fn clamp(&self, min: f32, max: f32) -> Self {
        Self {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
            z: self.z.clamp(min, max),
        }
    }

    /// Returns vector with powed values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector3};
    /// let a = Vector3::new(1.0, 2.0, 4.0);
    /// assert_eq!(a.powf(2.0), Vector3::new(1.0, 4.0, 16.0));
    /// ```
    #[inline]
    pub fn powf(&self, var: f32) -> Self {
        Self {
            x: self.x.powf(var),
            y: self.y.powf(var),
            z: self.z.powf(var),
        }
    }

    /// Returns vector with powed values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector3};
    /// let a = Vector3::new(1.0, 2.0, 4.0);
    /// assert_eq!(a.pow(2), Vector3::new(1.0, 4.0, 16.0));
    /// ```
    #[inline]
    pub fn pow(&self, var: i32) -> Self {
        self.powf(var as f32)
    }

    /// Returns vector with min values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(2.0, 1.0, 3.0);
    /// let b = Vector3::new(4.0, -1.0, 9.0);
    /// let c = Vector3::new(2.0, -1.0, 3.0);
    /// assert_eq!(a.min(&b), c);
    /// ```
    #[inline]
    pub fn min(&self, other: &Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    /// Returns vector with max values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(2.0, 1.0, 3.0);
    /// let b = Vector3::new(4.0, -1.0, 9.0);
    /// let c = Vector3::new(4.0, 1.0, 9.0);
    /// assert_eq!(a.max(&b), c);
    /// ```
    #[inline]
    pub fn max(&self, other: &Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    /// Returns vector with turncated values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(2.2, 1.9, 3.0);
    /// let b = Vector3::new(2.0, 1.0, 3.0);
    /// assert_eq!(a.trunc(), b);
    /// ```
    #[inline]
    pub fn trunc(&self) -> Self {
        Self {
            x: self.x.trunc(),
            y: self.y.trunc(),
            z: self.z.trunc(),
        }
    }

    /// Negates vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(12.0, -3.0, 4.0);
    /// let b = Vector3::new(-12.0, 3.0, -4.0);
    /// assert_eq!(a.neg(), b);
    /// ```
    #[inline]
    pub fn neg(&self) -> Self {
        Self {
            x: self.x.neg(),
            y: self.y.neg(),
            z: self.z.neg(),
        }
    }

    /// Flips a vector at current vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(1.0, 2.0, 4.0);
    /// let other = Vector3::new(3.0, 2.0, 4.0);
    /// assert_eq!(a.flip(other), Vector3::new(-1.0, 2.0, 4.0));
    /// ```
    #[inline]
    pub fn flip(&self, other: Self) -> Self {
        Self {
            x: self.x + (self.x - other.x),
            y: self.y + (self.y - other.y),
            z: self.z + (self.z - other.z),
        }
    }

    /// Returns dot product of this with another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(12.0, -3.0, 4.0);
    /// let b = Vector3::new(3.0, 3.0, 3.0);
    /// assert_eq!(a.dot(b), 39.0);
    /// ```
    #[inline]
    pub fn dot(&self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Returns length squared of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(12.0, -3.0, 4.0);
    /// assert_eq!(a.length_squared(), 169.0);
    /// ```
    #[inline]
    pub fn length_squared(&self) -> f32 {
        self.dot(*self)
    }

    /// Returns length of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(12.0, -3.0, 4.0);
    /// assert_eq!(a.length(), 13.0);
    /// ```
    #[inline]
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Checks if vector is normalized.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(12.0, -3.0, 4.0);
    /// let b = a.normalize();
    /// assert!(!a.is_normalized());
    /// assert!(b.is_normalized());
    /// ```
    #[inline]
    pub fn is_normalized(&self) -> bool {
        let len = self.length();
        const MIN: f32 = 1.0f32 - 1.0e-6;
        const MAX: f32 = 1.0f32 + 1.0e-6;
        //len >= MIN && len <= MAX
        (MIN..=MAX).contains(&len)
    }

    /// Returns length of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(12.0, -3.0, 4.0);
    /// assert_eq!(a.length(), 13.0);
    /// ```
    #[inline]
    pub fn magnitude(&self) -> f32 {
        self.length()
    }

    /// Returns sum of vector attrs.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(12.0, -3.0, 4.0);
    /// assert_eq!(a.sum(), 13.0);
    /// ```
    #[inline]
    pub fn sum(&self) -> f32 {
        self.x + self.y + self.z
    }

    /// Returns this distance squared to another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(12.0, -3.0, 4.0);
    /// let b = Vector3::new(-1.0, 3.0, 4.0);
    /// assert_eq!(a.distance_to_squared(b), 205.0);
    /// ```
    #[inline]
    pub fn distance_to_squared(&self, other: Self) -> f32 {
        (*self - other).length_squared()
    }

    /// Returns this distance squared to another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(12.0, -3.0, 4.0);
    /// let b = Vector3::new(-1.0, 3.0, 4.0);
    /// assert_eq!(a.distance_to(b), 14.3178215);
    /// ```
    #[inline]
    pub fn distance_to(&self, other: Self) -> f32 {
        self.distance_to_squared(other).sqrt()
    }

    /// Returns angle between this and another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(12.0, -3.0, 4.0);
    /// let b = Vector3::new(-1.0, 3.0, 4.0);
    /// assert_eq!(a.angle(b), 1.6462973);
    /// ```
    #[inline]
    pub fn angle(&self, other: Self) -> f32 {
        (self.dot(other) / (self.length_squared() * other.length_squared()).sqrt()).acos()
    }

    /// Vectors have to be normalized
    /// Returns angle between this and another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(12.0, -3.0, 4.0);
    /// let b = Vector3::new(-1.0, 3.0, 4.0);
    /// assert_eq!(a.normalize().angle(b.normalize()), 1.6462973);
    /// ```
    #[inline]
    pub fn angle_normalized(&self, other: Self) -> f32 {
        cgt_assert!(self.is_normalized());
        cgt_assert!(other.is_normalized());
        if self.dot(other) >= 0.0f32 {
            return 2.0 * (other-*self).length() / 2.0;
        }
        PI-2.0 * (other.neg()-*self).length() / 2.0
    }

    /// Returns normalized vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(2.0, -4.0, 21.0);
    /// let b = Vector3::new(0.09314928, -0.18629856, 0.97806746);
    /// assert_eq!(a.normalize(), b);
    /// ```
    /// TODO: Handle 0. Division
    #[inline]
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len <= 0.0001 {
            Self::ZERO
        } else {
            *self / self.length()
        }
    }

    /// Retuns this vectors cross product with given vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(-3.0, 21.0, 4.0);
    /// let b = Vector3::new(-1.0, 3.0, 4.0);
    /// let c = Vector3::new(72.0, 8.0, 12.0);
    /// assert_eq!(a.cross(b), c);
    /// ```
    #[inline]
    pub fn cross(&self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Returns this vector projected on another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(24.0, -5.0, 4.0);
    /// let b = Vector3::new(-1.0, 3.0, 4.0);
    /// let c = Vector3::new(0.88461536, -2.653846, -3.5384614);
    /// assert_eq!(a.project(b), c);
    /// ```
    #[inline]
    pub fn project(&self, other: Self) -> Self {
        other * (self.dot(other) / other.length_squared())
    }

    /// Returns projection of vector onto plane normal by substracting
    /// the component of u which is orthogonal to the plane from u
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(1.0, 1.0, 1.0);
    /// let b = Vector3::new(0.0, 1.0, 0.0);
    /// let c = Vector3::new(1.0, 0.0, 1.0);
    /// assert_eq!(a.orthogonal_projection(b), c);
    /// ```
    #[inline]
    pub fn orthogonal_projection(&self, normal: Self) -> Self {
        *self - normal * (self.dot(normal) / normal.length_squared())
    }

    /// Divide into equal parts, returns bisection vector
    /// # Example:
    /// ```
    /// use cgt_math::{Vector3, F32Utilities};
    /// let a = Vector3::new(1.0, 3.0, 1.0);
    /// let b = Vector3::new(2.0, 1.0, -2.0);
    /// let c = Vector3::new(3.0, -1.0, -2.0);
    /// let res = Vector3 { x: 0.39970168, y: -0.79940337, z: -0.44854525 };
    /// assert_eq!(a.bisect(b, c).fround(4), res.fround(4));
    /// ```
    #[inline]
    pub fn bisect(self, a: Self, b: Self) -> Self {
        ((a-self).normalize() + (b-a).normalize()).normalize()
    }

    /// Returns vector reflected from a plane defined by given normal.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(-1.0, 0.0, 2.0);
    /// let b = Vector3::new(0.0, 0.0, 1.0);
    /// assert_eq!(a.reflect(b), Vector3::new(-1.0, 0.0, -2.0));
    /// ```
    #[inline]
    pub fn reflect(&self, other: Self) -> Self {
        *self - other * (self.dot(other) * 2.0)
    }

    /// Returns this vector slid along plane defined by the given normal.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(2.0, -4.0, 21.0);
    /// let b = Vector3::new(-1.0, 0.0, 1.0);
    /// let c = Vector3::new(21.0, -4.0, 2.0);
    /// assert_eq!(a.slide(b), c);
    /// ```
    #[inline]
    pub fn slide(&self, other: Self) -> Self {
        *self - other * self.dot(other)
    }

    /// Returns any orthogonal to this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(24.0, -5.0, 4.0);
    /// assert_eq!(a.dot(a.orthogonal()), 0.0);
    /// ```
    #[inline]
    pub fn orthogonal(&self) -> Self {
        // http://lolengine.net/blog/2013/09/21/picking-orthogonal-vector-combing-coconuts
        let v = self.abs();
        if v.x > v.z {
            Self::new(-self.y, self.x, 0.0)
        } else {
            Self::new(0.0, -self.z, self.y)
        }
    }

    /// Returns any perpendicular to this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(24.0, -5.0, 4.0);
    /// assert_eq!(a.dot(a.perpendicular()), 0.0);
    /// ```
    #[inline]
    pub fn perpendicular(&self) -> Self {
        if self.x != 0.0 {
            Self::new(-self.y/self.x, 1.0, 0.0)
        }
        else if self.y != 0.0 {
            Self::new(0.0, -self.z/self.y, 1.0)
        }
        else {
            Self::new(1.0, 0.0, -self.x/self.z)
        }
    }

    /// Returns the inverse of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::{Vector3, F32Utilities};
    /// let a = Vector3::new(42.0, 1.0, 3.0).inverse();
    /// let b = Vector3::new(0.023809524, 1.0, 0.33333334);
    /// assert_eq!(a.fround(4), b.fround(4));
    /// ```
    #[inline]
    pub fn inverse(&self) -> Self {
        Self::new(1.0 / self.x, 1.0 / self.y, 1.0 / self.z)
    }

    /// Returns linear interpolation vector. T between [0, 1].
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(1.0, 0.0, 3.0);
    /// let b = Vector3::new(2.0, 0.0, 0.0);
    /// assert_eq!(a.interpolate(b, 0.5), Vector3::new(1.5, 0.0, 1.5));
    /// ```
    #[inline]
    pub fn interpolate(&self, rhs: Self, t: f32) -> Self {
        const S: f32 = 1.0;
        *self * (S-t) + rhs * t
    }

    /// Returns cubic interpolation vector. T between [0, 1].
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(1.0, 0.0, 3.0);
    /// let b = Vector3::new(2.0, 0.0, 0.0);
    /// let c = Vector3::new(1.0, 1.0, 1.0);
    /// let w = Vector3::new(-1.0, 2.0, 1.0);
    /// assert_eq!(a.interpolate_cubic(b, c, w), Vector3::new(4.0, 1.0, -2.0));
    /// ```
    #[inline]
    pub fn interpolate_cubic(&self, v1: Self, v2: Self, w: Self) -> Self {
        Self {
            x: self.x * w.x + v1.x * w.y + v2.x * w.z,
            y: self.y * w.x + v1.y * w.y + v2.y * w.z,
            z: self.z * w.x + v1.z * w.y + v2.z * w.z,
        }
    }

    /// Returns center of two points.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(1.0, 0.0, 3.0);
    /// let b = Vector3::new(2.0, 0.0, 0.0);
    /// assert_eq!(a.center(b), Vector3::new(1.5, 0.0, 1.5));
    /// ```
    #[inline]
    pub fn center(&self, other: Self) -> Self {
        Self {
            x: (self.x + other.x)*0.5f32,
            y: (self.y + other.y)*0.5f32,
            z: (self.z + other.z)*0.5f32,
        }
    }

    /// Returns center of three points.
    /// # Example:
    /// ```
    /// use cgt_math::Vector3;
    /// let a = Vector3::new(3.0, 0.0, 3.0);
    /// let b = Vector3::new(2.0, 0.0, 0.0);
    /// let c = Vector3::new(-2.0, 0.0, 3.0);
    /// assert_eq!(a.center_of_three(b, c), Vector3::new(1.0, 0.0, 2.0));
    /// ```
    #[inline]
    pub fn center_of_three(&self, v1: Self, v2: Self) -> Self {
        Self {
            x: (self.x + v1.x + v2.x)/3.0f32,
            y: (self.y + v1.y + v2.y)/3.0f32,
            z: (self.z + v1.z + v2.z)/3.0f32,
        }
    }

}

impl Add for Vector3 {
    type Output = Vector3;
    fn add(self, other: Self) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Add<f32> for Vector3 {
    type Output = Vector3;
    fn add(self, other: f32) -> Self::Output {
        Self {
            x: self.x + other,
            y: self.y + other,
            z: self.z + other,
        }
    }
}

impl AddAssign for Vector3 {
    fn add_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Vector3 {
    type Output = Vector3;
    fn sub(self, other: Self) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Sub<f32> for Vector3 {
    type Output = Vector3;
    fn sub(self, other: f32) -> Self::Output {
        Vector3 {
            x: self.x - other,
            y: self.y - other,
            z: self.z - other,
        }
    }
}

impl SubAssign for Vector3 {
    fn sub_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul<f32> for Vector3 {
    type Output = Self;
    fn mul(self, val: f32) -> Self::Output {
        Vector3 {
            x: self.x * val,
            y: self.y * val,
            z: self.z * val,
        }
    }
}

impl Mul<Quaternion> for Vector3 {
    type Output = Self;
    fn mul(self, quat: Quaternion) -> Self::Output {
        let q = &quat.q;
        let u = Vector3::new(q.x, q.y, q.z);
        let s = q.w;

        u*u.dot(self)*2.0f32 + self*(s*s-u.dot(u)) + u.cross(self)*2.0f32*s
    }
}

impl Mul<Vector3> for Vector3 {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        Vector3 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl MulAssign for Vector3 {
    fn mul_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl Div<f32> for Vector3 {
    type Output = Vector3;
    fn div(self, val: f32) -> Self::Output {
        Vector3 {
            x: self.x / val,
            y: self.y / val,
            z: self.z / val,
        }
    }
}

impl Div<Vector3> for Vector3 {
    type Output = Vector3;
    fn div(self, other: Self) -> Self::Output {
        Vector3 {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
        }
    }
}

impl DivAssign for Vector3 {
    fn div_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
        }
    }
}

impl Neg for Vector3 {
    type Output = Vector3;
    fn neg(self) -> Self::Output {
        Self {
            x: self.x * -1.0,
            y: self.y * -1.0,
            z: self.z * -1.0,
        }
    }
}

impl PartialEq for Vector3 {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}

impl Index<usize> for Vector3 {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index Error: {}", index),
        }
    }
}

impl IndexMut<usize> for Vector3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index Error: {}", index),
        }
    }
}
