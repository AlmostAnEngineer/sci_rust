use num_traits::{Bounded, FromPrimitive, Num};
use rand::distributions::uniform::SampleUniform;
use crate::ndarray::NDArray;

impl<T: Num + Copy + FromPrimitive> NDArray<T>
where
    T: Num + Copy + FromPrimitive + Bounded + PartialOrd + PartialEq + SampleUniform + Clone,
    {
        pub fn shape_equal(a: &NDArray<T>, b: &NDArray<T>) -> bool {
            a._shape == b._shape
        }
    }