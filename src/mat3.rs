//use crate::vector::{Vector3};

pub type Row = [f32; 4];

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    m: [Row; 4],
}

impl Matrix {
    pub fn new(r1: [f32; 4], r2: [f32; 4], r3: [f32; 4], r4: [f32; 4]) -> Self {
        Self {
            m: [r1, r2, r3, r4]
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RotationMatrix {
    m: [[f32; 3]; 3],
}


#[allow(dead_code)]
fn main() {
    let matrix = Matrix::new(
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    );
    println!("{:?}", matrix)
}
