extern crate cgt_math;
use cgt_math::{Vector3, Face, Circle};

#[cfg(test)]
mod geometry_tests {
    use crate::{Vector3, Face, Circle};

    #[test]
    fn impl_test() {
        // normal
        let face = Face::new([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [0, 1, 2]);
        let norm = face.normal();
        assert_eq!(norm, Vector3::new(0.0, -1.0, 0.0));

        // distance
        let point = Vector3::new(0.0, 1.0, 0.0);
        let dist = face.distance(point);
        assert_eq!(dist.abs(), 1.0);

        // to array
        let (vertices, connections) = face.to_array();
        assert_eq!((vertices, connections), ([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [0, 1, 2]));

        // projection
        let proj_vec = Vector3::new(1.0, 1.0, 0.0);
        let proj = face.project(proj_vec);
        assert_eq!(proj, Vector3::new(0.0f32, 1.0, 0.0f32));

        // reflection
        let refl_vec = Vector3::new(0.0, -1.0, 0.5);
        let refl = face.reflect(refl_vec);
        assert_eq!(refl, Vector3::new(0.0f32, 1.0, 0.5));
    }

    #[test]
    fn impl_circ() {
        let c1 = Circle::circle_from_angle(Vector3::new(0.0, 0.0, 0.0), 1.0, 0.0, 12);
        let c2 = Circle::circle_from_uv(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 1.0),
            Vector3::new(1.0, 1.0, 0.0),
            1.0,
            12,
        );
    }

}
