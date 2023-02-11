use std::f32;

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