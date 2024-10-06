use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct NDArray<T> {
    _data: Rc<RefCell<Vec<T>>>,
    _shape: Rc<RefCell<Vec<usize>>>,
    _strides: Rc<RefCell<Vec<usize>>>
}

pub mod ndarray;
pub mod operators;
pub mod traits;
pub mod utils;
pub mod display;
pub mod iterator;


#[cfg(test)]
mod tests;
