extern crate cgt_math;
use cgt_math::{Vector3, Face, Points};

#[cfg(test)]
mod geometry_tests {
    use crate::{Vector3, Face, Points};

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
        assert_eq!(proj, Vector3::new(1.0, 0.0f32, 0.0f32));

        // reflection
        let refl_vec = Vector3::new(0.0, -1.0, 0.5);
        let refl = face.reflect(refl_vec);
        assert_eq!(refl, Vector3::new(0.0f32, 1.0, 0.5));
    }

    #[test]
    fn impl_points() {
        // create circle
        let c1 = Points::circle(Vector3::new(0.0, 0.0, 0.0), 1.0, 4);
        let arr1 = c1.to_array::<4>();
        assert_eq!(arr1,[[1.0, 0.0, 0.0], [-4.371139e-8, 0.0, 1.0], [-1.0, 0.0, -8.742278e-8], [1.1924881e-8, 0.0, -1.0]]);

        // create circle from 2 vectors
        let center = Vector3::new(0.0, 0.0, 0.0);
        let v = Vector3::new(1.0, 0.0, 0.0);
        let u = Vector3::new(0.0, 1.0, 0.0);
        let c2 = Points::circle_from_uv(center, u, v.cross(u), 1.0, 6);
        let arr2 = c2.to_array::<6>();
        assert_eq!(arr2, [[0.0, 1.0, 0.0], [0.0, 0.49999997, 0.86602545], [0.0, -0.50000006, 0.8660254], [0.0, -1.0, -8.742278e-8], [0.0, -0.4999999, -0.86602545], [0.0, 0.4999999, -0.86602545]]);

        // create line from - to
        let x = Vector3::new(0.0, 0.0, 0.0);
        let y = Vector3::new(4.0, 4.0, 4.0);
        let line = Points::line_from_uv(x, y, 5);
        assert_eq!(line.to_array::<5>(), [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]);

        // check if close
        let close = line.closest_to(Vector3::new(2.1, 2.5, 1.5));
        assert_eq!(close, Vector3::new(2.0, 2.0, 2.0));

        // archemedian
        // let arche = Points::spiral_archemedian(6.0, 24.0, 1.0, 0.0, 0.2, true);
    }
}

