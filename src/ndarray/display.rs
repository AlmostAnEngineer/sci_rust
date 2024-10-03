use std::fmt::Display;
use crate::ndarray::NDArray;
use crate::ndarray::traits::NDArrayBounds;

impl<T> Display for NDArray<T>
where 
T: NDArrayBounds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[not supported]")
    }
}

