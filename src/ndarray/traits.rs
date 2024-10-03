use num::{Num, FromPrimitive};
use num_traits::Bounded;
use std::cmp::PartialOrd;
use rand::distributions::uniform::SampleUniform;

pub trait NDArrayBounds: Num + FromPrimitive + Copy + Bounded + PartialOrd + SampleUniform  {}
impl<T> NDArrayBounds for T where T: Num + FromPrimitive + Copy + Bounded + PartialOrd + SampleUniform {}
