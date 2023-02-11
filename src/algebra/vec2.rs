use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign, Index, IndexMut};
use std::f32::consts::PI;
use crate::{F32Utilities};

#[derive(Copy, Clone, Debug)]
pub struct Vector2 {
    pub x: f32,
    pub y: f32,
}

impl Vector2 {
    pub const ZERO: Self = Self::new(0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 1.0);
    pub const X: Self = Self::new(1.0, 0.0);
    pub const Y: Self = Self::new(0.0, 1.0);
    pub const NEG_X: Self = Self::new(-1.0, 0.0);
    pub const NEG_Y: Self = Self::new(0.0, -1.0);

    pub const INF: Self = Self::new(f32::INFINITY, f32::INFINITY);
    pub const NAN: Self = Self::new(f32::NAN, f32::NAN);
    pub const EPSILON: Self = Self::new(f32::EPSILON, f32::EPSILON);

    /// Create vector.
    /// # Example
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(0.0, 0.0);
    /// assert_eq!(a, Vector2::ZERO);
    /// ```
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Creates a new vector from an array.
    /// # Example
    /// ```
    /// use cgt_math::Vector2;
    /// let vec = Vector2::from_array([1.0, 1.0]);
    /// assert_eq!(vec, Vector2::ONE);
    /// ```
    #[inline]
    pub const fn from_array(a: [f32; 2]) -> Self {
        Self::new(a[0], a[1])
    }

    /// Creates array from vector
    /// # Example
    /// ```
    /// use cgt_math::Vector2;
    /// let vec = Vector2::from_array([1.0, 1.0]);
    /// assert_eq!(vec.to_array(), [1.0, 1.0]);
    /// ```
    #[inline]
    pub const fn to_array(&self) -> [f32; 2] {
        [self.x, self.y]
    }

    /// Returns if any vector component is nan.
    /// # Example
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(0.0, f32::NAN);
    /// assert!(a.is_nan());
    /// ```
    pub fn is_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() 
    }

    /// Returns if any vector component is infinte.
    /// # Example
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::INF;
    /// assert!(a.is_infinite());
    /// ```
    pub fn is_infinite(&self) -> bool {
        self.x.is_infinite() || self.y.is_infinite()
    }

    /// Returns if any vector componet is not finite.
    /// # Example
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(1.0, f32::INFINITY);
    /// let b = Vector2::new(1.0, f32::NAN);
    /// let c = Vector2::new(1.0, 1.0);
    /// assert!(!a.is_finite());
    /// assert!(!b.is_finite());
    /// assert!(c.is_finite());
    /// ```
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Returns vector with absolute values.
    /// # Example
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(-1.2, -4.0);
    /// assert_eq!(a.abs(), Vector2::new(1.2, 4.0));
    /// ```
    #[inline]
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }

    /// Returns vector with ceiled values.
    /// # Example
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(-1.2, 1.4);
    /// assert_eq!(a.ceil(), Vector2::new(-1.0, 2.0));
    /// ```
    #[inline]
    pub fn ceil(&self) -> Self {
        Self {
            x: self.x.ceil(),
            y: self.y.ceil(),
        }
    }

    /// Returns vector with floored values.
    /// # Example
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(-1.2, 1.4);
    /// assert_eq!(a.floor(), Vector2::new(-2.0, 1.0));
    /// ```
    #[inline]
    pub fn floor(&self) -> Self {
        Self {
            x: self.x.floor(),
            y: self.y.floor(),
        }
    }

    /// Returns vector with sinus of vector values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector2, F32Utilities};
    /// use std::f32::consts::PI;
    /// let a = Vector2::new(PI/2.0, -PI/2.0);
    /// let b = Vector2::new(1.0, -1.0);
    /// assert_eq!(a.sin().fround(4), b.fround(4));
    /// ```
    #[inline]
    pub fn sin(&self) -> Self {
        Self {
            x: self.x.sin(),
            y: self.y.sin(),
        }
    }

    /// Returns vector with inverse sinus of vector values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector2, F32Utilities};
    /// use std::f32::consts::PI;
    /// let a = Vector2::new(1.0, -1.0);
    /// let b = Vector2::new(PI/2.0, -PI/2.0);
    /// assert_eq!(a.asin().fround(4), b.fround(4));
    /// ```
    #[inline]
    pub fn asin(&self) -> Self {
        Self {
            x: self.x.asin(),
            y: self.y.asin(),
        }
    }

    /// Returns vector with cosinus of vector values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector2, F32Utilities};
    /// use std::f32::consts::PI;
    /// let a = Vector2::new(PI, 2.0*PI);
    /// let b = Vector2::new(-1.0, 1.0);
    /// assert_eq!(a.cos().fround(4), b.fround(4));
    /// ```
    #[inline]
    pub fn cos(&self) -> Self {
        Self {
            x: self.x.cos(),
            y: self.y.cos(),
        }
    }

    /// Returns vector with inverse cosinus of vector values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector2, F32Utilities};
    /// use std::f32::consts::PI;
    /// let a = Vector2::new(-1.0, 0.0);
    /// let b = Vector2::new(PI, PI/2.0);
    /// assert_eq!(a.acos().fround(4), b.fround(4));
    /// ```
    #[inline]
    pub fn acos(&self) -> Self {
        Self {
            x: self.x.acos(),
            y: self.y.acos(),
        }
    }

    /// Returns vector with rounded values.
    /// # Example
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(-1.411, 1.1451);
    /// assert_eq!(a.round(), Vector2::new(-1.0, 1.0));
    /// ```
    #[inline]
    pub fn round(&self) -> Self {
        Self {
            x: self.x.round(),
            y: self.y.round(),
        }
    }

    /// Returns vector with rounded values by k.
    /// # Example
    /// ```
    /// use cgt_math::{F32Utilities, Vector2};
    /// let a = Vector2::new(-1.411, 1.1451);
    /// assert_eq!(a.fround(2), Vector2::new(-1.41, 1.14));
    /// ```
    pub fn fround(&self, k: u32) -> Self {
        Self {
            x: self.x.fround(k),
            y: self.y.fround(k),
        }
    }

    /// Returns vector with clamped values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector2};
    /// let a = Vector2::new(-1.411, 1.1451);
    /// assert_eq!(a.clamp(-1.0, 1.0), Vector2::new(-1.0, 1.0));
    /// ```
    #[inline]
    pub fn clamp(&self, min: f32, max: f32) -> Self {
        Self {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
        }
    }

    /// Returns vector with powed values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector2};
    /// let a = Vector2::new(1.0, 4.0);
    /// assert_eq!(a.powf(2.0), Vector2::new(1.0, 16.0));
    /// ```
    #[inline]
    pub fn powf(&self, var: f32) -> Self {
        Self {
            x: self.x.powf(var),
            y: self.y.powf(var),
        }
    }

    /// Returns vector with powed values.
    /// # Example
    /// ```
    /// use cgt_math::{Vector2};
    /// let a = Vector2::new(1.0, 4.0);
    /// assert_eq!(a.pow(2), Vector2::new(1.0, 16.0));
    /// ```
    #[inline]
    pub fn pow(&self, var: i32) -> Self {
        self.powf(var as f32)
    }

    /// Returns vector with min values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(2.0, 3.0);
    /// let b = Vector2::new(4.0, 9.0);
    /// let c = Vector2::new(2.0, 3.0);
    /// assert_eq!(a.min(&b), c);
    /// ```
    #[inline]
    pub fn min(&self, other: &Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }

    /// Returns vector with max values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(2.0, 3.0);
    /// let b = Vector2::new(4.0, 9.0);
    /// let c = Vector2::new(4.0, 9.0);
    /// assert_eq!(a.max(&b), c);
    /// ```
    #[inline]
    pub fn max(&self, other: &Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
        }
    }

    /// Returns vector with turncated values.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(2.2, 3.0);
    /// let b = Vector2::new(2.0, 3.0);
    /// assert_eq!(a.trunc(), b);
    /// ```
    #[inline]
    pub fn trunc(&self) -> Self {
        Self {
            x: self.x.trunc(),
            y: self.y.trunc(),
        }
    }

    /// Negates vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(12.0, 4.0);
    /// let b = Vector2::new(-12.0, -4.0);
    /// assert_eq!(a.neg(), b);
    /// ```
    #[inline]
    pub fn neg(&self) -> Self {
        Self {
            x: self.x.neg(),
            y: self.y.neg(),
        }
    }

    /// Flips a vector at current vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(1.0, 4.0);
    /// let other = Vector2::new(3.0, 4.0);
    /// assert_eq!(a.flip(other), Vector2::new(-1.0, 4.0));
    /// ```
    #[inline]
    pub fn flip(&self, other: Self) -> Self {
        Self {
            x: self.x + (self.x - other.x),
            y: self.y + (self.y - other.y),
        }
    }

    /// Returns dot product of this with another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(12.0, -3.0);
    /// let b = Vector2::new(3.0, 3.0);
    /// assert_eq!(a.dot(b), 27.0);
    /// ```
    #[inline]
    pub fn dot(&self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Returns length squared of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(12.0, -3.0);
    /// assert_eq!(a.length_squared(), 153.0);
    /// ```
    #[inline]
    pub fn length_squared(&self) -> f32 {
        self.dot(*self)
    }

    /// Returns length of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(12.0, 4.0);
    /// assert_eq!(a.length(), 12.649111);
    /// ```
    #[inline]
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Checks if vector is normalized.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(12.0, 4.0);
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
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(12.0, 4.0);
    /// assert_eq!(a.length(), 12.649111);
    /// ```
    #[inline]
    pub fn magnitude(&self) -> f32 {
        self.length()
    }

    /// Returns sum of vector attrs.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(12.0, 4.0);
    /// assert_eq!(a.sum(), 16.0);
    /// ```
    #[inline]
    pub fn sum(&self) -> f32 {
        self.x + self.y
    }

    /// Returns this distance squared to another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(12.0, 4.0);
    /// let b = Vector2::new(-1.0, 4.0);
    /// assert_eq!(a.distance_to_squared(b), 169.0);
    /// ```
    #[inline]
    pub fn distance_to_squared(&self, other: Self) -> f32 {
        (*self - other).length_squared()
    }

    /// Returns this distance squared to another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(12.0, 4.0);
    /// let b = Vector2::new(-1.0, 4.0);
    /// assert_eq!(a.distance_to(b), 13.0);
    /// ```
    #[inline]
    pub fn distance_to(&self, other: Self) -> f32 {
        self.distance_to_squared(other).sqrt()
    }

    /// Returns angle between this and another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(12.0, 4.0);
    /// let b = Vector2::new(-1.0, 4.0);
    /// assert_eq!(a.angle(b), 1.4940244);
    /// ```
    #[inline]
    pub fn angle(&self, other: Self) -> f32 {
        (self.dot(other) / (self.length_squared() * other.length_squared()).sqrt()).acos()
    }

    /// Vectors have to be normalized
    /// Returns angle between this and another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(12.0, 4.0);
    /// let b = Vector2::new(-1.0, 4.0);
    /// assert_eq!(a.normalize().angle(b.normalize()), 1.4940244);
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
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(2.0, -4.0);
    /// let b = Vector2::new(0.4472136, -0.8944272);
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

    /// Returns this vector projected on another vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(24.0, 4.0);
    /// let b = Vector2::new(-1.0, 4.0);
    /// let c = Vector2::new(0.47058824, -1.882353);
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
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(2.25, 1.25);
    /// let b = Vector2::new(3.75, 0.25);
    /// let c = Vector2::new(-0.073009014, 1.0951327);
    /// assert_eq!(a.orthogonal_projection(b), c);
    /// ```
    #[inline]
    pub fn orthogonal_projection(&self, normal: Self) -> Self {
        *self - normal * (self.dot(normal) / normal.length_squared())
    }

    /// Divide into equal parts, returns bisection vector
    /// # Example:
    /// ```
    /// use cgt_math::{Vector2, F32Utilities};
    /// let a = Vector2::new(1.0, 1.0);
    /// let b = Vector2::new(2.0, -2.0);
    /// let c = Vector2::new(3.0, -2.0);
    /// let res = Vector2 { x: 0.8112, y: -0.5847 };
    /// assert_eq!(a.bisect(b, c).fround(4), res.fround(4));
    /// ```
    #[inline]
    pub fn bisect(self, a: Self, b: Self) -> Self {
        ((a-self).normalize() + (b-a).normalize()).normalize()
    }

    /// Returns vector reflected from a plane defined by given normal.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(-1.0, 2.0);
    /// let b = Vector2::new(0.0, 1.0);
    /// assert_eq!(a.reflect(b), Vector2::new(-1.0, -2.0));
    /// ```
    #[inline]
    pub fn reflect(&self, other: Self) -> Self {
        *self - other * (self.dot(other) * 2.0)
    }

    /// Returns this vector slid along plane defined by the given normal.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(2.0,  21.0);
    /// let b = Vector2::new(-1.0, 1.0);
    /// let c = Vector2::new(21.0, 2.0);
    /// assert_eq!(a.slide(b), c);
    /// ```
    #[inline]
    pub fn slide(&self, other: Self) -> Self {
        *self - other * self.dot(other)
    }

    /// Returns any orthogonal to this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(2.0,  21.0);
    /// assert_eq!(a.perpendicular(), Vector2::new(21.0, -2.0));
    /// ```
    #[inline]
    pub fn orthogonal(&self) -> Self {
        self.perpendicular()
    }

    /// Returns any perpendicular to this vector.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(2.0,  21.0);
    /// assert_eq!(a.perpendicular(), Vector2::new(21.0, -2.0));
    /// ```
    #[inline]
    pub fn perpendicular(&self) -> Self {
        if self.x > 0.0001f32 {
            // clockwise
            Vector2 { x: self.y, y: -self.x }
        } else {
            // counter clockwise
            Vector2 { x: -self.y, y: self.x }
        }
    }
    

    /// Returns the inverse of this vector.
    /// # Example:
    /// ```
    /// use cgt_math::{Vector2, F32Utilities};
    /// let a = Vector2::new(42.0, 3.0).inverse();
    /// let b = Vector2::new(0.023809524, 0.33333334);
    /// assert_eq!(a.fround(4), b.fround(4));
    /// ```
    #[inline]
    pub fn inverse(&self) -> Self {
        Self::new(1.0 / self.x, 1.0 / self.y)
    }

    /// Returns linear interpolation vector. T between [0, 1].
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(1.0, 3.0);
    /// let b = Vector2::new(2.0, 0.0);
    /// assert_eq!(a.interpolate(b, 0.5), Vector2::new(1.5, 1.5));
    /// ```
    #[inline]
    pub fn interpolate(&self, rhs: Self, t: f32) -> Self {
        const S: f32 = 1.0;
        *self * (S-t) + rhs * t
    }

    /// Returns cubic interpolation vector. T between [0, 1].
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(1.0, 3.0);
    /// let b = Vector2::new(2.0, 0.0);
    /// let w = Vector2::new(-1.0, 1.0);
    /// assert_eq!(a.interpolate_cubic(b, w), Vector2::new(1.0, -3.0));
    /// ```
    #[inline]
    pub fn interpolate_cubic(&self, v1: Self, w: Self) -> Self {
        *self * w.x + v1 * w.y
    }

    /// Returns center of two points.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(1.0, 3.0);
    /// let b = Vector2::new(2.0, 0.0);
    /// assert_eq!(a.center(b), Vector2::new(1.5, 1.5));
    /// ```
    #[inline]
    pub fn center(&self, other: Self) -> Self {
        (*self+other)*0.5f32
    }

    /// Returns center of three points.
    /// # Example:
    /// ```
    /// use cgt_math::Vector2;
    /// let a = Vector2::new(3.0, 3.0);
    /// let b = Vector2::new(2.0, 0.0);
    /// let c = Vector2::new(-2.0, 3.0);
    /// assert_eq!(a.center_of_three(b, c), Vector2::new(1.0, 2.0));
    /// ```
    #[inline]
    pub fn center_of_three(&self, v1: Self, v2: Self) -> Self {
        (*self+v1+v2)/3.0f32
    }

    /// Polynomial smoothing (x, y, factor)
    pub fn smin_polynomial(x: f32, y: f32, k: f32) -> Self {
        let m = x.smin_polynomial(y, k);
        let s = m*k*(1.0/3.0);
        if x < y {
            Self {
                x: x-s,
                y: m,
            }
        }
        else {
            Self {
                x: y-s,
                y: 1.0-m,
            }
        }
    }

    /// Cubic polynomial smoothing (x, y, factor)
    pub fn smin_polynomial_cubic(x: f32, y: f32, k: f32) -> Self {
        let m = x.smin_polynomial_cubic(y, k);
        let s = m*k*(1.0/3.0);
        if x < y {
            Self {
                x: x-s,
                y: m,
            }
        }
        else {
            Self {
                x: y-s,
                y: 1.0-m,
            }
        }
    }

    /// Generalization to any power n
    pub fn smin_polynomial_n(x: f32, y: f32, k: f32, n: f32) -> Self {
        let h = (k-(x-y).abs()).max(0.0)/k;
        let m = h.powf(n)*0.5;
        let s = m*k/n;
        if x < y {
            Self {
                x: x-s,
                y: m,
            }
        }
        else {
            Self {
                x: y-s,
                y: 1.0-m,
            }
        }
    }
}
impl Add for Vector2 {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}


impl AddAssign for Vector2 {
    fn add_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for Vector2 {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Sub<f32> for Vector2 {
    type Output = Self;
    fn sub(self, other: f32) -> Self::Output {
        Self {
            x: self.x - other,
            y: self.y - other,
        }
    }
}

impl SubAssign for Vector2 {
    fn sub_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Mul<f32> for Vector2 {
    type Output = Self;
    fn mul(self, val: f32) -> Self::Output {
        Self {
            x: self.x * val,
            y: self.y * val,
        }
    }
}

impl Mul<Vector2> for Vector2 {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
        }
    }
}

impl MulAssign for Vector2 {
    fn mul_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x * other.x,
            y: self.y * other.y,
        }
    }
}

impl Div<f32> for Vector2 {
    type Output = Self;
    fn div(self, val: f32) -> Self::Output {
        Self {
            x: self.x / val,
            y: self.y / val,
        }
    }
}

impl Div<Vector2> for Vector2 {
    type Output = Self;
    fn div(self, other: Self) -> Self::Output {
        Self {
            x: self.x / other.x,
            y: self.y / other.y,
        }
    }
}

impl DivAssign for Vector2 {
    fn div_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x / other.x,
            y: self.y / other.y,
        }
    }
}

impl Neg for Vector2 {
    type Output = Vector2;
    fn neg(self) -> Self::Output {
        Self {
            x: self.x * -1.0,
            y: self.y * -1.0,
        }
    }
}

impl PartialEq for Vector2 {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y 
    }
}

impl Index<usize> for Vector2 {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Index Error: {}", index),
        }
    }
}

impl IndexMut<usize> for Vector2 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Index Error: {}", index),
        }
    }
}
