extern crate cgt_math;
use cgt_math::{Euler, EulerOrder, Quaternion};


#[cfg(test)]
mod tests {
    use crate::Euler;
    use crate::EulerOrder;
    use crate::Quaternion;


    fn round_e(e: Euler) -> Euler {
        Euler::from_vec((e.v*1000.0).round()/1000.0)
    }

    #[test]
    fn test_to_quat() {
        let q = Quaternion::new(0.0, 0.0, 0.0, 1.0);
        let e = round_e(Euler::from_quat(q, EulerOrder::XYZ));
        assert_eq!(e, Euler::new(0.0, 0.0, 0.0));


        let q = Quaternion::new(1.0, 0.0, 0.0, 1.0);
        let e = round_e(Euler::from_quat(q, EulerOrder::XYZ));
        assert_eq!(e, Euler::new(1.571, 0.0, 0.0));

        let q = Quaternion::new(0.0, 1.0, 0.0, 1.0);
        let e = round_e(Euler::from_quat(q, EulerOrder::XYZ));
        assert_eq!(e, Euler::new(0.0, 1.571, 0.0));

        let q = Quaternion::new(0.0, 0.0, 1.0, 1.0);
        let e = round_e(Euler::from_quat(q, EulerOrder::XYZ));
        assert_eq!(e, Euler::new(0.0, 0.0, 1.571));
    }
}
