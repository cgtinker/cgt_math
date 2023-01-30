use std::f32;


// https://www.iquilezles.org/www/articles/smin/smin.htm
pub trait ProceduralOperators {
    type Output;

    fn interpolation_linear(self, a: Self, k: Self) -> Self::Output;
    fn smin_exponential(self, a: Self, k: Self) -> Self::Output;
    fn smin_power(self, a: Self, k: Self) -> Self::Output;
    fn smin_root(self, a: Self, k: Self) -> Self::Output;
    fn smin_polynomial(self, a: Self, k: Self) -> Self::Output;
    fn smin_polynomial_cubic(self, a: Self, k: Self) -> Self::Output;
}

impl ProceduralOperators for f32 {
    type Output = Self;

    // k = 0-1
    fn interpolation_linear(self, b: Self, k: Self) -> Self {
        self*(1.0-k)+b*k
    }

    /// Allows to process long lists of dists
    /// in any arbitrary order. 
    fn smin_exponential(self, b: Self, k: Self) -> Self {
        return -((-k*self).exp2()+(-k*b).exp2()).log2() / k;
    }

    /// default k=8
    fn smin_power(self, b: Self, k: Self) -> Self {
        let a = self.powf(k);
        let b = b.powf(k);
        return ((a*b)/(a+b)).powf(1.0/k);
    }

    /// default k=.01
    fn smin_root(self, b: Self, k: Self) -> Self {
        let h = self-b;
        return 0.5*((self+b)-(h*h+k).sqrt());
    }

    /// Polynomial ordering is dependant
    fn smin_polynomial(self, b: Self, k: Self) -> Self {
        let h = (k-(self-b).abs()).max(0.0)/k;
        return self.min(b) - h*h*k*(1.0/4.0);
    }

    /// Polynomial ordering is dependant
    fn smin_polynomial_cubic(self, b: Self, k: Self) -> Self {
        let h = (k-(self-b).abs()).max(0.0)/k;
        return self.min(b) - h*h*h*k*(1.0/6.0);
    }
}
