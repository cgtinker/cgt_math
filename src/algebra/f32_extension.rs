use std::f32;
use std::ops::{Add, Div, Mul, Sub};
use crate::{Vector2, Vector3, Vector4, Quaternion};

pub trait F32Utilities {
    type Output;

    fn fround(self, k: u32) -> Self::Output;
    // https://www.iquilezles.org/www/articles/smin/smin.htm
    fn interpolation_linear(self, a: Self, k: Self) -> Self::Output;
    fn smin_exponential(self, a: Self, k: Self) -> Self::Output;
    fn smin_power(self, a: Self, k: Self) -> Self::Output;
    fn smin_root(self, a: Self, k: Self) -> Self::Output;
    fn smin_polynomial(self, a: Self, k: Self) -> Self::Output;
    fn smin_polynomial_cubic(self, a: Self, k: Self) -> Self::Output;
}

impl F32Utilities for f32 {
    type Output = Self;
    /// Round number to digit k
    /// Example:
    /// ```
    /// use cgt_math::F32Utilities;
    /// let a: f32 = 1.3240914121;
    /// assert_eq!(a.fround(2), 1.32);
    /// ```
    fn fround(self, k: u32) -> Self {
        let p = 10_u32.pow(k) as f32;
        (self * p).trunc() / p
    }

    // k = 0-1
    fn interpolation_linear(self, b: Self, k: Self) -> Self {
        self*(1.0-k)+b*k
    }

    /// Allows to process long lists of dists
    /// in any arbitrary order. 
    fn smin_exponential(self, b: Self, k: Self) -> Self {
        -((-k*self).exp2()+(-k*b).exp2()).log2() / k
    }

    /// default k=8
    fn smin_power(self, b: Self, k: Self) -> Self {
        let a = self.powf(k);
        let b = b.powf(k);
        ((a*b)/(a+b)).powf(1.0/k)
    }

    /// default k=.01
    fn smin_root(self, b: Self, k: Self) -> Self {
        let h = self-b;
        0.5*((self+b)-(h*h+k).sqrt())
    }

    /// Polynomial ordering is dependant
    fn smin_polynomial(self, b: Self, k: Self) -> Self {
        let h = (k-(self-b).abs()).max(0.0)/k;
        self.min(b) - h*h*k*(1.0/4.0)
    }

    /// Polynomial ordering is dependant
    fn smin_polynomial_cubic(self, b: Self, k: Self) -> Self {
        let h = (k-(self-b).abs()).max(0.0)/k;
        self.min(b) - h*h*h*k*(1.0/6.0)
    }
}


/* Vector 2 */
impl Add<Vector2> for f32 {
    type Output = Vector2;
    fn add(self, rhs: Vector2) -> Self::Output {
        Vector2 {
            x: self + rhs.x,
            y: self + rhs.y,
        }
    }
}

impl Sub<Vector2> for f32 {
    type Output = Vector2;
    fn sub(self, rhs: Vector2) -> Self::Output {
        Vector2 {
            x: self - rhs.x,
            y: self - rhs.y,
        }
    }
}

impl Mul<Vector2> for f32 {
    type Output = Vector2;
    fn mul(self, rhs: Vector2) -> Self::Output {
        Vector2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}

impl Div<Vector2> for f32{
    type Output = Vector2;
    fn div(self, rhs: Vector2) -> Self::Output {
        Vector2 {
            x: self / rhs.x,
            y: self / rhs.y,
        }
    }
}

/* Vector3 */
impl Add<Vector3> for f32 {
    type Output = Vector3;
    fn add(self, rhs: Vector3) -> Self::Output {
        Vector3 {
            x: self + rhs.x,
            y: self + rhs.y,
            z: self + rhs.z,
        }
    }
}

impl Sub<Vector3> for f32 {
    type Output = Vector3;
    fn sub(self, rhs: Vector3) -> Self::Output {
        Vector3 {
            x: self - rhs.x,
            y: self - rhs.y,
            z: self - rhs.z,
        }
    }
}

impl Mul<Vector3> for f32 {
    type Output = Vector3;
    fn mul(self, rhs: Vector3) -> Self::Output {
        Vector3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl Div<Vector3> for f32{
    type Output = Vector3;
    fn div(self, rhs: Vector3) -> Self::Output {
        Vector3 {
            x: self / rhs.x,
            y: self / rhs.y,
            z: self / rhs.z,
        }
    }
}

/* Vector4 */
impl Add<Vector4> for f32 {
    type Output = Vector4;
    fn add(self, rhs: Vector4) -> Self::Output {
        Vector4 {
            x: self + rhs.x,
            y: self + rhs.y,
            z: self + rhs.z,
            w: self + rhs.w,
        }
    }
}

impl Sub<Vector4> for f32 {
    type Output = Vector4;
    fn sub(self, rhs: Vector4) -> Self::Output {
        Vector4 {
            x: self - rhs.x,
            y: self - rhs.y,
            z: self - rhs.z,
            w: self - rhs.w,
        }
    }
}

impl Mul<Vector4> for f32 {
    type Output = Vector4;
    fn mul(self, rhs: Vector4) -> Self::Output {
        Vector4 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
            w: self * rhs.w,
        }
    }
}

impl Div<Vector4> for f32{
    type Output = Vector4;
    fn div(self, rhs: Vector4) -> Self::Output {
        Vector4 {
            x: self / rhs.x,
            y: self / rhs.y,
            z: self / rhs.z,
            w: self / rhs.w,
        }
    }
}


/* Quaternion */
impl Add<Quaternion> for f32 {
    type Output = Quaternion;
    fn add(self, rhs: Quaternion) -> Self::Output {
        Quaternion {
            v: Vector4 {
                x: self + rhs.v.x,
                y: self + rhs.v.y,
                z: self + rhs.v.z,
                w: self + rhs.v.w,
            }
        }
    }
}

impl Sub<Quaternion> for f32 {
    type Output = Quaternion;
    fn sub(self, rhs: Quaternion) -> Self::Output {
        Quaternion {
            v: Vector4 {
                x: self - rhs.v.x,
                y: self - rhs.v.y,
                z: self - rhs.v.z,
                w: self - rhs.v.w,
            }
        }
    }
}


impl Mul<Quaternion> for f32 {
    type Output = Quaternion;
    fn mul(self, rhs: Quaternion) -> Self::Output {
        Quaternion {
            v: Vector4 {
                x: self * rhs.v.x,
                y: self * rhs.v.y,
                z: self * rhs.v.z,
                w: self * rhs.v.w,
            }
        }
    }
}

impl Div<Quaternion> for f32{
    type Output = Quaternion;
    fn div(self, rhs: Quaternion) -> Self::Output {
        Quaternion {
            v: Vector4 {
                x: self / rhs.v.x,
                y: self / rhs.v.y,
                z: self / rhs.v.z,
                w: self / rhs.v.w,
            }
        }
    }
}

