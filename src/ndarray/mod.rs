#[derive(Debug, Clone)]
pub struct NDArray<T>{
    _data: Vec<T>,
    _shape: Vec<usize>,
    _strides: Vec<usize>
}

pub mod ndarray;
pub mod ndarray_operators;
pub mod ndarray_traits;
pub mod ndarray_utils;


#[cfg(test)]
mod tests;
