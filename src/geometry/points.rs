use crate::Vector3;
use std::f32::consts::PI;
use std::convert::TryInto;
use std::ops::{Deref};

#[derive(Clone, Debug)]
pub struct Points {
    vec: Vec<Vector3>,
}


impl Points {
    pub fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
        let dx = (end - start) / (n-1) as f32;
        let mut res = vec![start; n];
        for i in 1..n {
            res[i] = res[i-1]+dx;
        }
        res
    }

    pub fn to_vector(&self) -> Vec<Vector3> {
        self.vec.clone()
    }

    pub fn to_array<const N: usize>(&self) -> [[f32; 3]; N] {
        let mut vec = Vec::with_capacity(self.vec.len());
        for i in 0..self.vec.len() {
            vec.push(self.vec[i].to_array());
        }
        vec.try_into().unwrap_or_else(|v: Vec<[f32; 3]>| panic!("Excepted len {} - was {}", v.len(), N))
    }

    pub fn to_varray<const N: usize>(&self) -> [Vector3; N] {
        self.vec.clone().try_into().unwrap_or_else(|v: Vec<Vector3>| panic!("Excepted len {} - was {}", v.len(), N))
    }

    pub fn to_box(&self) -> Box<[[f32; 3]]> {
        let mut vec = Vec::with_capacity(self.vec.capacity());
        for i in 0..self.vec.capacity() {
            vec.push(self.vec[i].to_array());
        }
        vec.into_boxed_slice()
    }

    pub fn to_vbox(&self) -> Box<[Vector3]> {
        self.vec.clone().into_boxed_slice()
    }

    /// Get closest by approx distance to point.
    pub fn closest_to(&self, point: Vector3) -> Vector3 {
        self.vec[self.closest_to_idx(point)]
    }


    /// Get closest idx by approx distance to point.
    pub fn closest_to_idx(&self, tar: Vector3) -> usize {
        let mut idx: usize = 0;
        let mut min: f32 = f32::MAX;
        for i in 0..self.vec.len() {
            let d = (self.vec[i]-tar).powf(2.0).sum();
            if d < min {
                min = d;
                idx = i;
            }
        }
        idx
    }

    pub fn line_from_uv(from: Vector3, to: Vector3, n: usize) -> Self {
        let lx: Vec<f32> = Self::linspace(from.x, to.x, n);
        let ly: Vec<f32> = Self::linspace(from.y, to.y, n);
        let lz: Vec<f32> = Self::linspace(from.z, to.z, n);

        let mut vec = Vec::with_capacity(n);
        for i in 0..n {
            let point = Vector3::new(lx[i], ly[i], lz[i]);
            vec.push(point);
        }
        Self { vec }
    }

    fn archemedian_spiral_properties(turns: f32, mut steps: f32, radius: f32, z_incr: f32, dif_radius: f32, clockwise: bool) -> 
        (f32, f32, f32, f32, f32, f32, f32, Vec<Vector3>) {

        let deg: f32 = 360.0*turns;
        steps *= turns;
        let z_scale: f32 = z_incr * turns;

        let mut max_phi: f32 = PI * deg / 180.0;
        let mut step_phi: f32 = max_phi / steps;

        if !clockwise {
            step_phi *= -1.0;
            max_phi *= -1.0;
        }

        let step_z: f32 = z_scale / (steps - 1.0);

        let mut verts: Vec<Vector3> = vec![Vector3::new(radius, 0.0, 0.0)];

        let cur_phi: f32 = 0.0;
        let cur_z: f32 = 0.0;
        let cur_rad: f32 = radius;

        let step_rad: f32 = dif_radius / (steps * 360.0 / deg);
        (cur_phi, max_phi, step_phi, cur_z, step_z, cur_rad, step_rad, verts)
    }

    // https://github.com/blender/blender-addons/
    // based on add_curve_spirals
    pub fn spiral_logarithmic(turns: f32, steps: f32, radius: f32, z_incr: f32, dif_radius: f32, force: f32, clockwise: bool) -> Self {
        let (mut cur_phi, max_phi, step_phi, mut cur_z, step_z, _cur_rad, _step_rad, mut verts) = Self::archemedian_spiral_properties(turns, steps, radius, z_incr, dif_radius, clockwise);
        while cur_phi.abs() <= max_phi.abs() {
            cur_phi += step_phi;
            cur_z += step_z;
            let cur_rad = radius * force.powf(cur_phi);
            verts.push(Vector3::new(cur_rad*cur_phi.cos(), cur_rad*cur_phi.sin(), cur_z));
        }
        Self { vec: verts }
    }

    // https://github.com/blender/blender-addons/
    // based on add_curve_spirals
    pub fn spiral_archemedian(turns: f32, steps: f32, radius: f32, z_incr: f32, dif_radius: f32, clockwise: bool) -> Self {
        let (mut cur_phi, max_phi, step_phi, mut cur_z, step_z, mut cur_rad, step_rad, mut verts) = Self::archemedian_spiral_properties(turns, steps, radius, z_incr, dif_radius, clockwise);
        while cur_phi.abs() <= max_phi.abs() {
            cur_phi += step_phi;
            cur_z += step_z;
            cur_rad += step_rad;
            verts.push(Vector3::new(cur_rad*cur_phi.cos(), cur_rad*cur_phi.sin(), cur_z));
        }
        Self { vec: verts }
    }

    // https://github.com/blender/blender-addons/
    // based on add_curve_spirals
    pub fn spiral_spherical(turns: f32, mut steps: f32, radius: f32,  clockwise: bool) -> Self {
        steps *= turns;
        let mut max_phi: f32 = 2.0*PI*turns;
        let mut step_phi: f32 = max_phi / ((2.0*PI) / steps);
        if !clockwise {
            max_phi *= -1.0;
            step_phi *= -1.0;
        }

        let step_theta: f32 = PI / (steps-1.0);
        let mut verts: Vec<Vector3> = Vec::new();
        verts.push(Vector3::new(0.0, 0.0, -radius));

        let mut cur_phi: f32 = 0.0;
        let mut cur_theta: f32 = -PI/2.0;

        while cur_phi.abs() <= max_phi.abs() {
            verts.push(Vector3::new(
                    radius * cur_theta.cos() * cur_phi.cos(),
                    radius * cur_theta.cos() * cur_phi.sin(),
                    radius * cur_theta.sin())
                );
            cur_theta += step_theta;
            cur_phi += step_phi;
        }
        Self { vec: verts }
    }

    pub fn arc(center: Vector3, radius: f32, from_angle: f32, to_angle: f32, n: usize) -> Self {
        let linspace: Vec<f32> = Self::linspace(from_angle, to_angle, n);
        let vec: Vec<Vector3> = linspace
            .iter()
            .take(n)
            .map(|theta| Vector3::new(
                    center.x+theta.cos()*radius,
                    center.y,
                    center.z+theta.sin()*radius))
            .collect();
        Self { vec }
    }

    pub fn circle(center: Vector3, radius: f32, n: usize) -> Self {
        let linspace: Vec<f32> = Self::linspace(0.0, 2.0*PI, n+1);
        let vec: Vec<Vector3> = linspace
            .iter()
            .take(n)
            .map(|theta| Vector3::new(
                    center.x+theta.cos()*radius,
                    center.y,
                    center.z+theta.sin()*radius))
            .collect();
        Self { vec }
    }

    /// U & V have to be normalized!
    /// TODO: Panic if not normalized
    pub fn circle_from_uv(center: Vector3, u: Vector3, v: Vector3, radius: f32, n: usize) -> Self {
        let linspace = Self::linspace(0.0, 2.0*PI, n+1);
        let mut vec = Vec::with_capacity(n);

        for theta in linspace.iter().take(n) {
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            let point = Vector3::new(
                center.x + radius * u.x * cos_t + radius * v.x * sin_t,
                center.y + radius * u.y * cos_t + radius * v.y * sin_t,
                center.z + radius * u.z * cos_t + radius * v.z * sin_t,
            );
            vec.push(point);
        }
        Self { vec }
    }
}

//impl Index<usize> for Points {
//    type Output = Vector3;
//    fn index(&self, index: usize) -> &Self::Output {
//        &self.vec[index]
//    }
//}

impl Deref for Points {
    type Target = Vec<Vector3>;
    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}
