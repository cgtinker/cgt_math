use crate::Vector3;
use::std::f32::consts::PI;

#[derive(Clone, Debug)]
pub struct Circle {
    points: Vec<Vector3>,
}


impl Circle {
    pub fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
        let dx = (end - start) / (n-1) as f32;
        let mut res = vec![start; n];
        for i in 1..n {
            res[i] = res[i-1]+dx;
        }
        res
    }

    pub fn from_angle(center: Vector3, radius: f32, angle: f32, points: usize) -> Self {
        let thetha: Vec<f32> = Self::linspace(0.0, 2.0*PI, points);
        let mut vec = Vec::with_capacity(points);
        for i in 0..points {
            let cos_t = thetha[i].cos();
            let sin_t = thetha[i].sin();

            let point = Vector3::new(
                center.x + cos_t * angle.cos() * radius,
                center.y + cos_t * angle.sin() * radius,
                center.z + sin_t * radius,
            );
            vec.push(point);
        }
        Self { points: vec }
    }

    /// U & V have to be normalized!
    /// TODO: Panic if not normalized
    pub fn from_uv(center: Vector3, u: Vector3, v: Vector3, radius: f32, points: usize) -> Self {
        let v = u.cross(v);;

        let thetha = Self::linspace(0.0, 2.0*PI, points);
        let mut vec = Vec::with_capacity(points);

        for i in 0..points {
            let cos_t = thetha[i].cos();
            let sin_t = thetha[i].sin();

            let point = Vector3::new(
                center.x + radius * u.x * cos_t + radius * v.x * sin_t,
                center.y + radius * u.y * cos_t + radius * v.y * sin_t,
                center.z + radius * u.z * cos_t + radius * v.z * sin_t,
            );
            vec.push(point);
        }
        Self { points: vec }
    }

    pub fn from_vector(center: Vector3, vec: Vector3, radius: f32, points: usize) -> Self {
        // searching for perpendicualar vector
        if vec.x != 0.0f32:
            let u = Vector3::new(-vec.y / vec.x, 1.0, 0.0);
        else if vec.y != 0.0f32:
            let u = Vector3::new(0.0, -vec.z / vec.y, 1.0);
        else:
            let u = Vector3::new(1.0, 0.0, -vec.x / vec.z);

        along_vectors(center, u, v, radius, points)
    }
}
