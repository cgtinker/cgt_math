use super::vec4;
use vec4::Vector4;

#[allow(dead_code)]
use std::{ops::*};

#[derive(Copy, Clone, Debug)]
pub struct Mat4x4 {
    pub cx: Vector4,
    pub cy: Vector4,
    pub cz: Vector4,
    pub cw: Vector4,
}

impl Mat4x4 {
    pub const ZERO: Self = Self::from_vecs(Vector4::ZERO, Vector4::ZERO, Vector4::ZERO, Vector4::ZERO);
    pub const NAN: Self = Self::from_vecs(Vector4::NAN, Vector4::NAN, Vector4::NAN, Vector4::NAN);
    pub const IDENTITY: Self = Self::from_vecs(Vector4::X, Vector4::Y, Vector4::Z, Vector4::W);

    /// Create new Matrix from arrays.
    /// # Example
    /// ````
    /// use cgt_math::{Vector4, Mat4x4};
    /// let m1 = Mat4x4::new(
    ///     [0.0, 0.1, 0.2, 0.3],
    ///     [1.0, 1.1, 1.2, 1.3],
    ///     [2.0, 2.1, 2.2, 2.3],
    ///     [3.0, 3.1, 3.2, 3.3],
    /// );
    /// let m2 = Mat4x4::from_vecs(
    ///     Vector4::new(0.0, 0.1, 0.2, 0.3),
    ///     Vector4::new(1.0, 1.1, 1.2, 1.3),
    ///     Vector4::new(2.0, 2.1, 2.2, 2.3),
    ///     Vector4::new(3.0, 3.1, 3.2, 3.3),
    /// );
    /// assert_eq!(m1, m2);
    /// ````
    pub const fn new(
        x_axis: [f32; 4],
        y_axis: [f32; 4],
        z_axis: [f32; 4],
        w_axis: [f32; 4],
    ) -> Self {
        Self::from_vecs(
            Vector4::from_array(x_axis),
            Vector4::from_array(y_axis),
            Vector4::from_array(z_axis),
            Vector4::from_array(w_axis),
        )
    }

    pub const fn from_vecs(
        x_axis: Vector4,
        y_axis: Vector4,
        z_axis: Vector4,
        w_axis: Vector4,
    ) -> Self {
        Self {
            cx: x_axis,
            cy: y_axis,
            cz: z_axis,
            cw: w_axis,
        }
    }

    pub const fn to_array(&self) -> [[f32; 4]; 4] {
        [
            self.cx.to_array(),
            self.cy.to_array(),
            self.cz.to_array(),
            self.cw.to_array(),
        ]
    }
}

impl PartialEq for Mat4x4 {
    fn eq(&self, other: &Self) -> bool {
        self.cx == other.cx && self.cy == other.cy && self.cz == other.cz && self.cw == other.cw
    }
}

#[allow(dead_code)]
fn main() {
    let matrix = Mat4x4::ZERO;
    println!("{:?}", matrix);
    // println!("{:?}", matrix.transpone);

    let m1 = Mat4x4::new(
        [0.0, 0.1, 0.2, 0.3],
        [1.0, 1.1, 1.2, 1.3],
        [2.0, 2.1, 2.2, 2.3],
        [3.0, 3.1, 3.2, 3.3],
    );
    let m2 = Mat4x4::from_vecs(
        Vector4::new(0.0, 0.1, 0.2, 0.3),
        Vector4::new(1.0, 1.1, 1.2, 1.3),
        Vector4::new(2.0, 2.1, 2.2, 2.3),
        Vector4::new(3.0, 3.1, 3.2, 3.3),
    );
    assert_eq!(m1, m2);
    let arr = m2.to_array();
    println!("{:?}", arr);
}
