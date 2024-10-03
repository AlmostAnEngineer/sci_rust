#[derive(Debug, Clone)]
pub struct NDArray<T>{
    _data: Vec<T>,
    _shape: Vec<usize>,
    _strides: Vec<usize>
}

pub mod ndarray;
pub mod operators;
pub mod traits;
pub mod utils;
pub mod display;


#[cfg(test)]
mod tests;
