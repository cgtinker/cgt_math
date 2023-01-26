extern crate cgt_math;
use cgt_math::Vector3;

#[cfg(test)]
mod tests {
    use crate::Vector3;

    #[test]
    fn test_vector_creation() {
        let nan = Vector3::NAN;
        assert!(nan.is_nan());
        let zero = Vector3::ZERO;
        assert_eq!(zero, Vector3::new(0.0, 0.0, 0.0));
        let one = Vector3::ONE;
        assert_eq!(one, Vector3::from_array([1.0, 1.0, 1.0]));
        let x = Vector3::X;
        assert_eq!(x, Vector3::from_array([1.0, 0.0, 0.0]));
        let y = Vector3::Y;
        assert_eq!(y, Vector3::from_array([0.0, 1.0, 0.0]));
        let z = Vector3::Z;
        assert_eq!(z, Vector3::from_array([0.0, 0.0, 1.0]));
        let nx = Vector3::NEG_X;
        assert_eq!(nx, Vector3::from_array([-1.0, 0.0, 0.0]));
        let ny = Vector3::NEG_Y;
        assert_eq!(ny, Vector3::from_array([0.0, -1.0, 0.0]));
        let nz = Vector3::NEG_Z;
        assert_eq!(nz, Vector3::from_array([0.0, 0.0, -1.0]));
        let inf = Vector3::INF;
        assert!(inf.is_infinite());
        assert!(!inf.is_finite());
        assert!(!nan.is_finite());
        assert!(one.is_finite());
        assert_eq!(z.to_array(), [0.0, 0.0, 1.0]);
    }
}
