//use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAs>
//use std::f32::consts::PI;
use crate::ProceduralOperators;

#[derive(Copy, Clone, Debug)]
pub struct Vector2 {
    x: f32,
    y: f32,
}

impl Vector2 {
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
    pub fn smin_polynomialN(x: f32, y: f32, k: f32, n: f32) -> Self {
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
