#[derive(Debug)]
pub struct NDArray<T>{
    _data: Vec<T>,
    _shape: Vec<usize>,
    _strides: Vec<usize>
}

mod ndarray;
mod ndarray_operators;

#[cfg(test)]
mod tests;
mod utils;
