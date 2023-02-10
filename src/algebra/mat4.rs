/* this script includes various martrix fns from: https://github.com/bitshifter/glam-rs f32/mat4.rs
 */
use crate::{Vector3, Vector4, Quaternion, RotationMatrix};
use std::ops::{Sub};
//use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Copy, Clone, Debug)]
pub struct Mat4x4 {
    pub x: Vector4,
    pub y: Vector4,
    pub z: Vector4,
    pub w: Vector4,
}

impl Mat4x4 {
    pub const ZERO: Self = Self::from_vecs(Vector4::ZERO, Vector4::ZERO, Vector4::ZERO, Vector4::ZERO);
    pub const NAN: Self = Self::from_vecs(Vector4::NAN, Vector4::NAN, Vector4::NAN, Vector4::NAN);
    pub const IDENTITY: Self = Self::from_vecs(Vector4::X, Vector4::Y, Vector4::Z, Vector4::W);

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x0: f32, x1: f32, x2: f32, x3: f32,
        y0: f32, y1: f32, y2: f32, y3: f32,
        z0: f32, z1: f32, z2: f32, z3: f32,
        w0: f32, w1: f32, w2: f32, w3: f32) -> Self {
        Self::from_vecs(
            Vector4::new(x0, x1, x2, x3),
            Vector4::new(y0, y1, y2, y3),
            Vector4::new(z0, z1, z2, z3),
            Vector4::new(w0, w1, w2, w3),
        )
    }

    pub fn from_array(arr: [[f32; 4]; 4]) -> Self {
        Self::from_vecs(
            Vector4::from_array(arr[0]),
            Vector4::from_array(arr[1]),
            Vector4::from_array(arr[2]),
            Vector4::from_array(arr[3]),
        )
    }

    pub const fn from_vecs(
        x_axis: Vector4,
        y_axis: Vector4,
        z_axis: Vector4,
        w_axis: Vector4,
    ) -> Self {
        Self {
            x: x_axis,
            y: y_axis,
            z: z_axis,
            w: w_axis,
        }
    }

    pub const fn to_array(&self) -> [[f32; 4]; 4] {
        [
            self.x.to_array(),
            self.y.to_array(),
            self.z.to_array(),
            self.w.to_array(),
        ]
    }

    // TODO: move to quat
    #[inline]
    fn quat_to_axes(q: Quaternion) -> (Vector4, Vector4, Vector4) {
        let (x, y, z, w) = (q.v.x, q.v.y, q.v.z, q.v.w);
        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;
        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        let x_axis = Vector4::new(1.0 - (yy + zz), xy + wz, xz - wy, 0.0);
        let y_axis = Vector4::new(xy - wz, 1.0 - (xx + zz), yz + wx, 0.0);
        let z_axis = Vector4::new(xz + wy, yz - wx, 1.0 - (xx + yy), 0.0);
        (x_axis, y_axis, z_axis)
    }

    ///
    /// Will panic if `rotation` is not normalized when `glam_assert` is enabled.
    //#[inline]
    //pub fn from_scale_rotation_translation(scale: Vector3, rotation: Quaternion, translation: Vector3) -> Self {
    //    let (x_axis, y_axis, z_axis) = Self::quat_to_axes(rotation);
    //    Self::from_vecs(
    //        x_axis.mul(scale.x),
    //        y_axis.mul(scale.y),
    //        z_axis.mul(scale.z),
    //        Vector4::from((translation, 1.0)),
    //    )
    //}

    #[inline]
    pub fn from_quat(rotation: Quaternion) -> Self {
        let (x_axis, y_axis, z_axis) = Self::quat_to_axes(rotation);
        Self::from_vecs(x_axis, y_axis, z_axis, Vector4::W)
    }

    #[inline]
    pub fn to_mat3(&self) -> RotationMatrix {
        RotationMatrix::from_vecs(
            Vector3::new(self.x.x, self.x.y, self.x.z),
            Vector3::new(self.y.x, self.y.y, self.y.z),
            Vector3::new(self.z.x, self.z.y, self.z.z),
        )
    }

    #[inline]
    pub fn from_mat3(m: RotationMatrix) -> Self {
        Self::from_vecs(
            Vector4::new(m.x[0], m.x[1], m.x[2], 0.0),
            Vector4::new(m.y[0], m.y[1], m.y[2], 0.0),
            Vector4::new(m.z[0], m.z[1], m.z[2], 0.0),
            Vector4::W,
        )
    }

    pub fn from_translation(translation: Vector3) -> Self {
        Self::from_vecs(
            Vector4::X,
            Vector4::Y,
            Vector4::Z,
            Vector4::new(translation.x, translation.y, translation.z, 1.0),
        )
    }

    /// Will panic if `axis` is not normalized when `glam_assert` is enabled.
    #[inline]
    pub fn from_axis_angle(axis: Vector3, angle: f32) -> Self {
        let rm = RotationMatrix::from_axis_angle(axis, angle);
        Self::from_mat3(rm)
    }

     #[inline]
    pub fn from_rotation_x(angle: f32) -> Self {
        let rm = RotationMatrix::from_rotation_x(angle);
        Self::from_mat3(rm)
    }

    /// Creates an affine transformation matrix containing a 3D rotation around the y axis of
    /// `angle` (in radians).
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    pub fn from_rotation_y(angle: f32) -> Self {
        let rm = RotationMatrix::from_rotation_y(angle);
        Self::from_mat3(rm)
    }

    /// Creates an affine transformation matrix containing a 3D rotation around the z axis of
    /// `angle` (in radians).
    ///
    /// The resulting matrix can be used to transform 3D points and vectors. See
    /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
    #[inline]
    pub fn from_rotation_z(angle: f32) -> Self {
        let rm = RotationMatrix::from_rotation_z(angle);
        Self::from_mat3(rm)
    }

    #[inline]
    pub fn from_scale(scale: Vector3) -> Self {
        // Do not panic as long as any component is non-zero
        Self::from_vecs(
            Vector4::new(scale.x, 0.0, 0.0, 0.0),
            Vector4::new(0.0, scale.y, 0.0, 0.0),
            Vector4::new(0.0, 0.0, scale.z, 0.0),
            Vector4::W,
        )
    }

     /// Returns the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 3.
    #[inline]
    pub fn col(&self, index: usize) -> Vector4 {
        match index {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            3 => self.w,
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns a mutable reference to the matrix column for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 3.
    #[inline]
    pub fn col_mut(&mut self, index: usize) -> &mut Vector4 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns the matrix row for the given `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than 3.
    #[inline]
    pub fn row(&self, index: usize) -> Vector4 {
        match index {
            0 => Vector4::new(self.x.x, self.y.x, self.z.x, self.w.x),
            1 => Vector4::new(self.x.y, self.y.y, self.z.y, self.w.y),
            2 => Vector4::new(self.x.z, self.y.z, self.z.z, self.w.z),
            3 => Vector4::new(self.x.w, self.y.w, self.z.w, self.w.w),
            _ => panic!("index out of bounds"),
        }
    }

    /// Returns `true` if, and only if, all elements are finite.
    /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite()
            && self.y.is_finite()
            && self.z.is_finite()
            && self.w.is_finite()
    }

    /// Returns `true` if any elements are `NaN`.
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }

       /// Returns the transpose of `self`.
    #[must_use]
    #[inline]
    pub fn transpose(&self) -> Self {
        // Based on https://github.com/microsoft/DirectXMath `XMMatrixTranspose`
        let v0 = self.x.merge_xy(self.z);
        let v1 = self.y.merge_xy(self.w);
        let v2 = self.x.merge_zw(self.z);
        let v3 = self.y.merge_zw(self.w);
        Self::from_vecs(
            v0.merge_xy(v1),
            v0.merge_zw(v1),
            v2.merge_xy(v3),
            v2.merge_zw(v3),
        )
    }

     /// Creates a left-handed view matrix using a camera position, an up direction, and a facing
    /// direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    #[inline]
    pub fn look_to_lh(eye: Vector3, dir: Vector3, up: Vector3) -> Self {
        Self::look_to_rh(eye, -dir, up)
    }

    /// Creates a right-handed view matrix using a camera position, an up direction, and a facing
    /// direction.
    ///
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    #[inline]
    pub fn look_to_rh(eye: Vector3, dir: Vector3, up: Vector3) -> Self {
        let f = dir.normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        Self::from_vecs(
            Vector4::new(s.x, u.x, -f.x, 0.0),
            Vector4::new(s.y, u.y, -f.y, 0.0),
            Vector4::new(s.z, u.z, -f.z, 0.0),
            Vector4::new(-eye.dot(s), -eye.dot(u), eye.dot(f), 1.0),
        )
    }

    /// Creates a left-handed view matrix using a camera position, an up direction, and a focal
    /// point.
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
    ///
    /// # Panics
    /// Will panic if `up` is not normalized when `glam_assert` is enabled.
    #[inline]
    pub fn look_at_lh(eye: Vector3, center: Vector3, up: Vector3) -> Self {
        Self::look_to_lh(eye, center.sub(eye), up)
    }

    /// Creates a right-handed view matrix using a camera position, an up direction, and a focal
    /// point.
    /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
    ///
    /// # Panics
    /// Will panic if `up` is not normalized when `glam_assert` is enabled.
    #[inline]
    pub fn look_at_rh(eye: Vector3, center: Vector3, up: Vector3) -> Self {
        Self::look_to_rh(eye, center.sub(eye), up)
    }

    /// Creates a right-handed perspective projection matrix with [-1,1] depth range.
    /// This is the same as the OpenGL `gluPerspective` function.
    /// See <https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml>
    #[inline]
    pub fn perspective_rh_gl(
        fov_y_radians: f32,
        aspect_ratio: f32,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        let inv_length = 1.0 / (z_near - z_far);
        let f = 1.0 / (0.5 * fov_y_radians).tan();
        let a = f / aspect_ratio;
        let b = (z_near + z_far) * inv_length;
        let c = (2.0 * z_near * z_far) * inv_length;
        Self::from_vecs(
            Vector4::new(a, 0.0, 0.0, 0.0),
            Vector4::new(0.0, f, 0.0, 0.0),
            Vector4::new(0.0, 0.0, b, -1.0),
            Vector4::new(0.0, 0.0, c, 0.0),
        )
    }

    /// Creates a left-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// # Panics
    ///
    /// Will panic if `z_near` or `z_far` are less than or equal to zero when `glam_assert` is
    /// enabled.
    #[inline]
    pub fn perspective_lh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
        let (sin_fov, cos_fov) = (0.5 * fov_y_radians).sin_cos();
        let h = cos_fov / sin_fov;
        let w = h / aspect_ratio;
        let r = z_far / (z_far - z_near);
        Self::from_vecs(
            Vector4::new(w, 0.0, 0.0, 0.0),
            Vector4::new(0.0, h, 0.0, 0.0),
            Vector4::new(0.0, 0.0, r, 1.0),
            Vector4::new(0.0, 0.0, -r * z_near, 0.0),
        )
    }


    /// Creates a right-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// # Panics
    ///
    /// Will panic if `z_near` or `z_far` are less than or equal to zero when `glam_assert` is
    /// enabled.
    #[inline]
    pub fn perspective_rh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
        let (sin_fov, cos_fov) = (0.5 * fov_y_radians).sin_cos();
        let h = cos_fov / sin_fov;
        let w = h / aspect_ratio;
        let r = z_far / (z_near - z_far);
        Self::from_vecs(
            Vector4::new(w, 0.0, 0.0, 0.0),
            Vector4::new(0.0, h, 0.0, 0.0),
            Vector4::new(0.0, 0.0, r, -1.0),
            Vector4::new(0.0, 0.0, r * z_near, 0.0),
        )
    }

    /// Creates an infinite left-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// # Panics
    ///
    /// Will panic if `z_near` is less than or equal to zero when `glam_assert` is enabled.
    #[inline]
    pub fn perspective_infinite_lh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32) -> Self {
        let (sin_fov, cos_fov) = (0.5 * fov_y_radians).sin_cos();
        let h = cos_fov / sin_fov;
        let w = h / aspect_ratio;
        Self::from_vecs(
            Vector4::new(w, 0.0, 0.0, 0.0),
            Vector4::new(0.0, h, 0.0, 0.0),
            Vector4::new(0.0, 0.0, 1.0, 1.0),
            Vector4::new(0.0, 0.0, -z_near, 0.0),
        )
    }

       /// Creates an infinite left-handed perspective projection matrix with `[0,1]` depth range.
    ///
    /// # Panics
    ///
    /// Will panic if `z_near` is less than or equal to zero when `glam_assert` is enabled.
    #[inline]
    pub fn perspective_infinite_reverse_lh(
        fov_y_radians: f32,
        aspect_ratio: f32,
        z_near: f32,
    ) -> Self {
        let (sin_fov, cos_fov) = (0.5 * fov_y_radians).sin_cos();
        let h = cos_fov / sin_fov;
        let w = h / aspect_ratio;
        Self::from_vecs(
            Vector4::new(w, 0.0, 0.0, 0.0),
            Vector4::new(0.0, h, 0.0, 0.0),
            Vector4::new(0.0, 0.0, 0.0, 1.0),
            Vector4::new(0.0, 0.0, z_near, 0.0),
        )
    }

    /// Creates an infinite right-handed perspective projection matrix with
    /// `[0,1]` depth range.
    #[inline]
    pub fn perspective_infinite_rh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32) -> Self {
        let f = 1.0 / (0.5 * fov_y_radians).tan();
        Self::from_vecs(
            Vector4::new(f / aspect_ratio, 0.0, 0.0, 0.0),
            Vector4::new(0.0, f, 0.0, 0.0),
            Vector4::new(0.0, 0.0, -1.0, -1.0),
            Vector4::new(0.0, 0.0, -z_near, 0.0),
        )
    }

        /// Creates an infinite reverse right-handed perspective projection matrix
    /// with `[0,1]` depth range.
    #[inline]
    pub fn perspective_infinite_reverse_rh(
        fov_y_radians: f32,
        aspect_ratio: f32,
        z_near: f32,
    ) -> Self {
        let f = 1.0 / (0.5 * fov_y_radians).tan();
        Self::from_vecs(
            Vector4::new(f / aspect_ratio, 0.0, 0.0, 0.0),
            Vector4::new(0.0, f, 0.0, 0.0),
            Vector4::new(0.0, 0.0, 0.0, -1.0),
            Vector4::new(0.0, 0.0, z_near, 0.0),
        )
    }

        /// Creates a right-handed orthographic projection matrix with `[-1,1]` depth
    /// range.  This is the same as the OpenGL `glOrtho` function in OpenGL.
    /// See
    /// <https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glOrtho.xml>
    #[inline]
    pub fn orthographic_rh_gl(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let a = 2.0 / (right - left);
        let b = 2.0 / (top - bottom);
        let c = -2.0 / (far - near);
        let tx = -(right + left) / (right - left);
        let ty = -(top + bottom) / (top - bottom);
        let tz = -(far + near) / (far - near);

        Self::from_vecs(
            Vector4::new(a, 0.0, 0.0, 0.0),
            Vector4::new(0.0, b, 0.0, 0.0),
            Vector4::new(0.0, 0.0, c, 0.0),
            Vector4::new(tx, ty, tz, 1.0),
        )
    }

    /// Creates a left-handed orthographic projection matrix with `[0,1]` depth range.
    #[inline]
    pub fn orthographic_lh(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let rcp_width = 1.0 / (right - left);
        let rcp_height = 1.0 / (top - bottom);
        let r = 1.0 / (far - near);
        Self::from_vecs(
            Vector4::new(rcp_width + rcp_width, 0.0, 0.0, 0.0),
            Vector4::new(0.0, rcp_height + rcp_height, 0.0, 0.0),
            Vector4::new(0.0, 0.0, r, 0.0),
            Vector4::new(
                -(left + right) * rcp_width,
                -(top + bottom) * rcp_height,
                -r * near,
                1.0,
            ),
        )
    }

    /// Creates a right-handed orthographic projection matrix with `[0,1]` depth range.
    #[inline]
    pub fn orthographic_rh(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let rcp_width = 1.0 / (right - left);
        let rcp_height = 1.0 / (top - bottom);
        let r = 1.0 / (near - far);
        Self::from_vecs(
            Vector4::new(rcp_width + rcp_width, 0.0, 0.0, 0.0),
            Vector4::new(0.0, rcp_height + rcp_height, 0.0, 0.0),
            Vector4::new(0.0, 0.0, r, 0.0),
            Vector4::new(
                -(left + right) * rcp_width,
                -(top + bottom) * rcp_height,
                r * near,
                1.0,
            ),
        )
    }
    /*
    /// Transforms the given 3D vector as a point, applying perspective correction.
    ///
    /// This is the equivalent of multiplying the 3D vector as a 4D vector where `w` is `1.0`.
    /// The perspective divide is performed meaning the resulting 3D vector is divided by `w`.
    ///
    /// This method assumes that `self` contains a projective transform.
    #[inline]
    pub fn project_point3(&self, rhs: Vector3) -> Vector3 {
        let mut res = self.x_axis.mul(rhs.x);
        res = self.y.mul(rhs.y).add(res);
        res = self.z.mul(rhs.z).add(res);
        res = self.w.add(res);
        res = res.mul(res.wwww().recip());
        res.xyz()
    }

    /// Transforms the give 3D vector as a direction.
    ///
    /// This is the equivalent of multiplying the 3D vector as a 4D vector where `w` is
    /// `0.0`.
    ///
    /// This method assumes that `self` contains a valid affine transform.
    ///
    /// # Panics
    ///
    /// Will panic if the 3rd row of `self` is not `(0, 0, 0, 1)` when `glam_assert` is enabled.
    #[inline]
    pub fn transform_vector3(&self, rhs: Vector3) -> Vector3 {
        let mut res = self.x.mul(rhs.x);
        res = self.y.mul(rhs.y).add(res);
        res = self.z.mul(rhs.z).add(res);
        res.xyz()
    }

    /// Returns true if the absolute difference of all elements between `self` and `rhs`
    /// is less than or equal to `max_abs_diff`.
    ///
    /// This can be used to compare if two matrices contain similar elements. It works best
    /// when comparing with a known value. The `max_abs_diff` that should be used used
    /// depends on the values being compared against.
    ///
    /// For more see
    /// [comparing floating point numbers](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/).
    #[inline]
    pub fn abs_diff_eq(&self, rhs: Self, max_abs_diff: f32) -> bool {
        self.x_axis.abs_diff_eq(rhs.x_axis, max_abs_diff)
            && self.y_axis.abs_diff_eq(rhs.y_axis, max_abs_diff)
            && self.z_axis.abs_diff_eq(rhs.z_axis, max_abs_diff)
            && self.w_axis.abs_diff_eq(rhs.w_axis, max_abs_diff)
    }
    */
}

impl PartialEq for Mat4x4 {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z && self.w == other.w
    }
}
