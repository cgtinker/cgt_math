extern crate cgt_math;
use cgt_math::{RotationMatrix, Quaternion, Vector3};
//use cgt_math::Quaternion;

#[cfg(test)]
mod tests {
    use crate::{RotationMatrix, Quaternion, Vector3};
    //use crate::Quaternion;
    //use crate::Vector3;

    #[test]
    fn new_rot_matrix() {
        let default = RotationMatrix::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0);
        let arr = RotationMatrix::from_array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]);
        assert_eq!(default, arr);
    }

    #[test]
    fn from_quat() {
        let q = Quaternion::new(1.0, 1.0, 0.0, 0.0);
        let m = RotationMatrix::from_quaternion(q.normalize());

        assert_eq!(m.round(), RotationMatrix::new(
                1.0, 0.0, 0.0,
                0.0, 0.0, 1.0,
                0.0, -1.0, 0.0));
    }


    #[test]
    fn from_axis_angle() {
        let v = Vector3::new(0.0, 0.0, 1.0);
        let m = RotationMatrix::from_axis_angle(v, 0.1);
        assert_eq!(m, RotationMatrix::new(
                0.9950042, 0.09983342, 0.0,
                -0.09983342, 0.9950042, 0.0, 
                0.0, 0.0, 1.0));
    }
}

