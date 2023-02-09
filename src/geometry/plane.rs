use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::{Vector3};

#[derive(Clone, Copy, Debug)]
pub struct Plane {
    v0: Vector3,
    v1: Vector3,
    v2: Vector3,
    connections: [u32; 3],
}

impl Plane {
    /// Create a new face.
    /// # Example
    /// ```
    /// use cgt_math::{Plane, Vector3};
    /// let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    /// let connections = [0, 1, 2];
    /// let face = Plane::new(vertices, connections);
    /// ```
    pub fn new(vertices: [[f32; 3]; 3], connections: [u32; 3]) -> Self {
        Self::from_vecs(
            Vector3::from_array(vertices[0]),
            Vector3::from_array(vertices[1]),
            Vector3::from_array(vertices[2]),
            connections,
        )
    }

    /// Create a new face from vectors.
    pub const fn from_vecs(v0: Vector3, v1: Vector3, v2: Vector3, connections: [u32; 3]) -> Self {
        Self {
            v0,
            v1,
            v2,
            connections,
        }
    }

    /// Returns vertices and connections as array.
    /// # Example
    /// ```
    /// use cgt_math::{Plane, Vector3};
    /// let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    /// let face = Plane::new(vertices, [0, 1, 2]);
    /// let (verts, connections) = face.to_array();
    /// assert_eq!((vertices, [0, 1, 2]), (verts, connections))
    /// ```
    pub const fn to_array(&self) -> ([[f32; 3]; 3], [u32; 3]) {
        ([self.v0.to_array(), self.v1.to_array(), self.v2.to_array()], self.connections)
    }

    /// Returns 'true' if all face point coords are finite.
    pub fn is_finite(self) -> bool {
        self.v0.is_finite() && self.v1.is_finite() && self.v2.is_finite()
    }

    /// Returns 'true' if any face point coord is infinite.
    pub fn is_infinite(self) -> bool {
        self.v0.is_infinite() || self.v1.is_infinite() || self.v2.is_infinite()
    }

    /// Returns 'true' if any face point coord is nan.
    pub fn is_nan(self) -> bool {
        self.v0.is_nan() || self.v1.is_nan() || self.v2.is_nan()
    }

    /// Returns normal from face.
    /// # Example
    /// ```
    /// use cgt_math::{Plane, Vector3};
    /// let face = Plane::new([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [0, 1, 2]);
    /// let norm = face.normal();
    /// assert_eq!(norm, Vector3::new(0.0, -1.0, 0.0));
    /// ```
    pub fn normal(self) -> Vector3 {
        (self.v1 - self.v0).cross(self.v2 - self.v0)
    }

    /// Returns distance of point to face as signed int.
    /// # Example
    /// ```
    /// use cgt_math::{Plane, Vector3};
    /// let face = Plane::new([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [0, 1, 2]);
    /// let p1 = Vector3::new(0.0, 1.0, 0.0);
    /// assert_eq!(face.distance(p1), -1.0);
    /// let p2 = Vector3::new(0.0, -1.0, 0.0);
    /// assert_eq!(face.distance(p2), 1.0);
    /// ```
    pub fn distance(&self, point: Vector3) -> f32 {
        (point - self.v0).dot(self.normal())
    }

    /// Projects input vector on face.
    /// # Example
    /// ```
    /// use cgt_math::{Plane, Vector3};
    /// let face = Plane::new([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [0, 1, 2]);
    /// let vec = Vector3::new(1.0, 1.0, 0.0);
    /// let proj = face.project(vec);
    /// assert_eq!(proj, Vector3::new(1.0, 0.0, 0.0));
    /// ```
    pub fn project(&self, vec: Vector3) -> Vector3 {
        vec.orthogonal_projection(self.normal())
    }

    /// Reflects input vector from face.
    /// # Example
    /// ```
    /// use cgt_math::{Plane, Vector3};
    /// let face = Plane::new([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [0, 1, 2]);
    /// let vec = Vector3::new(0.0, -1.0, 1.0);
    /// let proj = face.reflect(vec);
    /// assert_eq!(proj, Vector3::new(0.0, 1.0, 1.0));
    /// ```
    pub fn reflect(&self, vec: Vector3) -> Vector3 {
        vec.reflect(self.normal())
    }

    // Requires a different approach -> velocity vector
    // shouldn't slide along the normal.
    // /// Slides input vector on face normal.
    // /// # Example
    // /// ```
    // /// use cgt_math::{Plane, Vector3};
    // /// let face = Plane::new([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [0, 1, 2]);
    // /// let vec = Vector3::new(0.0, -1.0, 0.5);
    // /// let proj = face.slide(vec);
    // /// assert_eq!(proj, Vector3::new(0.0, 0.0, 0.5));
    // /// ```
    // pub fn slide(&self, vec: Vector3) -> Vector3 {
    //     vec.slide(self.normal())
    // }
}


impl Add<Vector3> for Plane {
    type Output = Plane;
    fn add(self, other: Vector3) -> Self::Output {
        Self {
            v0: self.v0 + other,
            v1: self.v1 + other,
            v2: self.v2 + other,
            connections: self.connections
        }
    }
}

impl AddAssign<Vector3> for Plane {
    fn add_assign(&mut self, other: Vector3) {
        *self = Self {
            v0: self.v0 + other,
            v1: self.v1 + other,
            v2: self.v2 + other,
            connections: self.connections
        }
    }
}

impl Sub<Vector3> for Plane {
    type Output = Plane;
    fn sub(self, other: Vector3) -> Self::Output {
        Self {
            v0: self.v0 - other,
            v1: self.v1 - other,
            v2: self.v2 - other,
            connections: self.connections
        }
    }
}

impl SubAssign<Vector3> for Plane {
    fn sub_assign(&mut self, other: Vector3) {
        *self = Self {
            v0: self.v0 - other,
            v1: self.v1 - other,
            v2: self.v2 - other,
            connections: self.connections
        }
    }
}

impl Mul<f32> for Plane {
    type Output = Plane;
    fn mul(self, val: f32) -> Self::Output {
        Self {
            v0: self.v0 * val,
            v1: self.v1 * val,
            v2: self.v2 * val,
            connections: self.connections
        }
    }
}

impl Mul<Vector3> for Plane {
    type Output = Plane;
    fn mul(self, other: Vector3) -> Self::Output {
        Self {
            v0: self.v0 * other,
            v1: self.v1 * other,
            v2: self.v2 * other,
            connections: self.connections
        }
    }
}

impl MulAssign<Vector3> for Plane {
    fn mul_assign(&mut self, other: Vector3) {
        *self = Self {
            v0: self.v0 * other,
            v1: self.v1 * other,
            v2: self.v2 * other,
            connections: self.connections
        }
    }
}

impl Div<f32> for Plane {
    type Output = Self;
    fn div(self, val: f32) -> Self::Output {
        Self {
            v0: self.v0 / val,
            v1: self.v1 / val,
            v2: self.v2 / val,
            connections: self.connections
        }
    }
}

impl Div<Vector3> for Plane {
    type Output = Self;
    fn div(self, other: Vector3) -> Self::Output {
        Self {
            v0: self.v0 / other,
            v1: self.v1 / other,
            v2: self.v2 / other,
            connections: self.connections
        }
    }
}

impl DivAssign<Vector3> for Plane {
    fn div_assign(&mut self, other: Vector3) {
        *self = Self {
            v0: self.v0 / other,
            v1: self.v1 / other,
            v2: self.v2 / other,
            connections: self.connections
        }
    }
}

impl Neg for Plane {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            v0: self.v0 * -1.0,
            v1: self.v1 * -1.0,
            v2: self.v2 * -1.0,
            connections: self.connections
        }
    }
}

// Doesn't compare connections as the comparison may be used to check for duplicate faces.
impl PartialEq for Plane {
    fn eq(&self, other: &Self) -> bool {
        self.v0 == other.v0 && self.v1 == other.v1 && self.v2 == other.v2
    }
}
