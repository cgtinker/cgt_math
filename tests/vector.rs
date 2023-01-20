/*
extern crate cgt_math;
use cgt_math::Vector3;

#[cfg(test)]
mod tests {
    use crate::Vector3;

    #[test]
    fn test_is_nan() {
        let a = Vector3::NAN;
        assert!(a.is_nan());
    }

    #[test]
    fn test_is_infinite() {
        let a = Vector3::INF;
        assert!(a.is_infinite());
    }

    #[test]
    fn test_abs() {
        let a = Vector3::new(-1.0, 0.0, 2.0);
        let b = Vector3::new(1.0, 0.0, 2.0);
        assert_eq!(a.abs(), b);
    }

    #[test]
    fn test_floor() {
        let a = Vector3::new(-1.3, 0.9, 2.5);
        let b = Vector3::new(-2.0, 0.0, 2.0);
        assert_eq!(a.floor(), b);
    }

    #[test]
    fn test_ceil() {
        let a = Vector3::new(-1.3, 0.9, 2.5);
        let b = Vector3::new(-1.0, 1.0, 3.0);
        assert_eq!(a.ceil(), b);
    }

    #[test]
    fn test_round() {
        let a = Vector3::new(-1.3, 0.9, 2.5);
        let b = Vector3::new(-1.0, 1.0, 3.0);
        assert_eq!(a.round(), b);
    }

    #[test]
    fn test_pow() {
        let a = Vector3::new(2.0, 1.0, 3.0);
        let b = Vector3::new(4.0, 1.0, 9.0);
        assert_eq!(a.powf(2.0), b);
    }

    #[test]
    fn test_clamp() {
        let a = Vector3::new(-1.3, 0.9, 2.5);
        let b = Vector3::new(-1.0, 0.9, 1.0);
        assert_eq!(a.clamp(-1.0, 1.0), b);
    }

    #[test]
    fn test_min() {
        let a = Vector3::new(-1.3, 0.9, 2.5);
        let b = Vector3::new(-1.0, 0.9, 1.0);
        assert_eq!(a.min(&b), Vector3::new(-1.3, 0.9, 1.0));
    }

    #[test]
    fn test_max() {
        let a = Vector3::new(-1.3, 0.9, 2.5);
        let b = Vector3::new(-1.0, 0.9, 1.0);
        assert_eq!(a.max(&b), Vector3::new(-1.0, 0.9, 2.5));
    }

    #[test]
    fn test_trunc() {
        let a = Vector3::new(-1.3, 0.9, 2.5);
        assert_eq!(a.trunc(), Vector3::new(-1.0, 0.0, 2.0));
    }

    #[test]
    fn test_length() {
        let a = Vector3::new(12.0, -3.0, 4.0);
        assert_eq!(a.length(), 13.0);
    }

    #[test]
    fn test_length_squared() {
        let a = Vector3::new(12.0, -3.0, 4.0);
        assert_eq!(a.length_squared(), 169.0);
    }

    #[test]
    fn test_dot() {
        let a = Vector3::new(12.0, -3.0, 4.0);
        let b = Vector3::new(3.0, 3.0, 3.0);
        assert_eq!(a.dot(b), 39.0);

        let c = Vector3::new(9.3, 1.5, 21.0);
        let d = Vector3::new(512.1, 10.2, 1.0);
        assert_eq!(c.dot(d), 4798.8296);
    }

    #[test]
    fn test_magnitude() {
        let a = Vector3::new(12.0, -3.0, 4.0);
        assert_eq!(a.magnitude(), 13.0);
    }

    #[test]
    fn test_sum() {
        let a = Vector3::new(12.0, -3.0, 4.0);
        assert_eq!(a.sum(), 13.0);
    }

    #[test]
    fn test_distance_to_squared() {
        let a = Vector3::new(12.0, -3.0, 4.0);
        let b = Vector3::new(-1.0, 3.0, 4.0);
        assert_eq!(a.distance_to_squared(b), 205.0);
    }

    #[test]
    fn test_distance_to() {
        let a = Vector3::new(12.0, -3.0, 4.0);
        let b = Vector3::new(-1.0, 3.0, 4.0);
        assert_eq!(a.distance_to(b), 14.3178215);
    }

    #[test]
    fn angle() {
        let a = Vector3::new(12.0, -3.0, 4.0);
        let b = Vector3::new(-1.0, 3.0, 4.0);
        assert_eq!(a.angle(b), 1.6462973);
    }

    #[test]
    fn test_normalize() {
        let a = Vector3::new(2.0, -4.0, 21.0);
        let b = Vector3::new(0.09314928, -0.18629856, 0.97806746);
        assert_eq!(a.normalize(), b);
    }

    #[test]
    fn test_cross() {
        let a = Vector3::new(-3.0, 21.0, 4.0);
        let b = Vector3::new(-1.0, 3.0, 4.0);
        let c = Vector3::new(72.0, 8.0, 12.0);
        assert_eq!(a.cross(b), c);
    }

    #[test]
    fn test_project() {
        let a = Vector3::new(24.0, -5.0, 4.0);
        let b = Vector3::new(-1.0, 3.0, 4.0);
        let c = Vector3::new(0.88461536, -2.653846, -3.5384614);
        assert_eq!(a.project(b), c);
    }
    #[test]
    fn test_reflect() {
        let a = Vector3::new(-1.0, 0.0, 2.0);
        let b = Vector3::new(0.0, 0.0, 1.0);
        assert_eq!(a.reflect(b), Vector3::new(-1.0, 0.0, -2.0));
    }

    #[test]
    fn test_slide() {
        let a = Vector3::new(2.0, -4.0, 21.0);
        let b = Vector3::new(-1.0, 0.0, 1.0);
        let c = Vector3::new(21.0, -4.0, 2.0);
        assert_eq!(a.slide(b), c);
    }

    #[test]
    fn test_orthogonal() {
        let a = Vector3::new(24.0, -5.0, 4.0);
        assert_eq!(a.dot(a.orthogonal()), 0.0)
    }

    #[test]
    fn test_inverse() {
        assert_eq!(
            Vector3::new(42.0, 1.0, 3.0).inverse(),
            Vector3::new(0.023809524, 1.0, 0.33333334)
        )
    }

    #[test]
    fn test_mul() {
        let a = Vector3::new(2.0, -2.0, 2.0);
        let b = Vector3::new(4.0, -4.0, 4.0);
        assert_eq!(a * 2.0, b)
    }

    #[test]
    fn test_vec_mul() {
        let a = Vector3::new(2.0, -2.0, 2.0);
        let b = Vector3::new(2.0, 2.0, 2.0);
        let c = Vector3::new(4.0, -4.0, 4.0);
        assert_eq!(a * b, c)
    }

    #[test]
    fn test_div() {
        let a = Vector3::new(2.0, -2.0, 2.0);
        let b = Vector3::new(1.0, -1.0, 1.0);
        assert_eq!(a / 2.0, b)
    }

    #[test]
    fn test_vec_div() {
        let a = Vector3::new(2.0, -2.0, 2.0);
        let b = Vector3::new(2.0, 2.0, 2.0);
        let c = Vector3::new(1.0, -1.0, 1.0);
        assert_eq!(a / b, c)
    }

    #[test]
    fn test_add() {
        let a = Vector3::new(2.0, -2.0, 2.0);
        let b = Vector3::new(1.0, -1.0, 1.0);
        let c = Vector3::new(3.0, -3.0, 3.0);
        assert_eq!(a + b, c)
    }

    #[test]
    fn test_sub() {
        let a = Vector3::new(2.0, -2.0, 2.0);
        let b = Vector3::new(1.0, -1.0, 1.0);
        let c = Vector3::new(1.0, -1.0, 1.0);
        assert_eq!(a - b, c)
    }

    #[test]
    fn test_negate() {
        assert_eq!(
            -Vector3::new(42.0, -1.0, 3.0),
            Vector3::new(42.0, -1.0, 3.0) * -1.0
        )
    }

    #[test]
    fn test_equality() {
        assert_eq!(Vector3::new(42.0, 1.0, 3.0), Vector3::new(42.0, 1.0, 3.0))
    }

    #[test]
    fn test_inequality() {
        assert_ne!(Vector3::new(42.10, 1.0, 3.0), Vector3::new(42.0, 1.0, 3.0))
    }

    #[test]
    fn test_index() {
        let a = Vector3::new(3.0, 3.0, 3.0);
        assert_eq!(a[0], 3.0);
    }
}
*/