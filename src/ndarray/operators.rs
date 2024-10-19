use std::ops::{Add, Sub, Mul, Div, Rem, AddAssign, SubAssign, MulAssign, DivAssign, RemAssign};
use std::cmp::PartialEq;
use crate::ndarray::traits::NDArrayBounds;
use crate::ndarray::NDArray;
use std::cell::RefCell;
use std::rc::Rc;

impl<T: NDArrayBounds> Add for &NDArray<T> {
    type Output = Option<NDArray<T>>;

    fn add(self, rhs: Self) -> Self::Output {
        if !NDArray::shape_equal(self, rhs) {
            return None;
        }
        let buffored_output = add_two_vectors(&self._data.borrow(), &rhs._data.borrow());
        let new_array = NDArray {
            _data: Rc::new(RefCell::new(buffored_output)),
            _shape: self._shape.clone(),
            _strides: self._strides.clone(),
        };
        Some(new_array)
    }
}

impl<T: NDArrayBounds> Sub for &NDArray<T> {
    type Output = Option<NDArray<T>>;

    fn sub(self, rhs: Self) -> Self::Output {
        if !NDArray::shape_equal(self, rhs) {
            return None;
        }
        let buffored_output = substract_two_vectors(&self._data.borrow(), &rhs._data.borrow());
        let new_array = NDArray {
            _data: Rc::new(RefCell::new(buffored_output)),
            _shape: self._shape.clone(),
            _strides: self._strides.clone(),
        };
        Some(new_array)
    }
}

impl<T: NDArrayBounds> Mul for &NDArray<T> {
    type Output = Option<NDArray<T>>;

    fn mul(self, rhs: Self) -> Self::Output {
        if !NDArray::shape_equal(self, rhs) {
            return None;
        }
        let buffored_output = mul_two_vectors(&self._data.borrow(), &rhs._data.borrow());
        let new_array = NDArray {
            _data: Rc::new(RefCell::new(buffored_output)),
            _shape: self._shape.clone(),
            _strides: self._strides.clone(),
        };
        Some(new_array)
    }
}

impl<T: NDArrayBounds> Div for &NDArray<T> {
    type Output = Option<NDArray<T>>;

    fn div(self, rhs: Self) -> Self::Output {
        if !NDArray::shape_equal(self, rhs) {
            return None;
        }
        let buffored_output = div_two_vectors(&self._data.borrow(), &rhs._data.borrow());
        let new_array = NDArray {
            _data: Rc::new(RefCell::new(buffored_output)),
            _shape: self._shape.clone(),
            _strides: self._strides.clone(),
        };
        Some(new_array)
    }
}

impl<T: NDArrayBounds> Rem for &NDArray<T> {
    type Output = Option<NDArray<T>>;

    fn rem(self, rhs: Self) -> Self::Output {
        if !NDArray::shape_equal(self, rhs) {
            return None;
        }
        let buffored_output = rem_two_vectors(&self._data.borrow(), &rhs._data.borrow());
        let new_array = NDArray {
            _data: Rc::new(RefCell::new(buffored_output)),
            _shape: self._shape.clone(),
            _strides: self._strides.clone(),
        };
        Some(new_array)
    }
}


impl<T: NDArrayBounds> PartialEq for NDArray<T> {
    fn eq(&self, other: &NDArray<T>) -> bool {
        if !NDArray::shape_equal(self, other) {
            return false;
        }
        self._data.borrow().iter().eq(other._data.borrow().iter())
    }
}


impl<T: NDArrayBounds> AddAssign for NDArray<T> {
    fn add_assign(&mut self, rhs: Self) {
        if !NDArray::shape_equal(self, &rhs) {
            panic!("cannot add assign");
        }
        let buffored_output = add_two_vectors(&self._data.borrow(), &rhs._data.borrow());
        *self._data.borrow_mut() = buffored_output;
    }
}

impl<T: NDArrayBounds> SubAssign for NDArray<T> {
    fn sub_assign(&mut self, rhs: Self) {
        if !NDArray::shape_equal(self, &rhs) {
            panic!("cannot sub assign");
        }
        let buffored_output = substract_two_vectors(&self._data.borrow(), &rhs._data.borrow());
        *self._data.borrow_mut() = buffored_output;
    }
}

impl<T: NDArrayBounds> MulAssign for NDArray<T> {
    fn mul_assign(&mut self, rhs: Self) {
        if !NDArray::shape_equal(self, &rhs) {
            panic!("cannot mul assign");
        }
        let buffored_output = mul_two_vectors(&self._data.borrow(), &rhs._data.borrow());
        *self._data.borrow_mut() = buffored_output;
    }
}

impl<T: NDArrayBounds> DivAssign for NDArray<T> {
    fn div_assign(&mut self, rhs: Self) {
        if !NDArray::shape_equal(self, &rhs) {
            panic!("cannot div assign");
        }
        let buffored_output = div_two_vectors(&self._data.borrow(), &rhs._data.borrow());
        *self._data.borrow_mut() = buffored_output;
    }
}

impl<T: NDArrayBounds> RemAssign for NDArray<T> {
    fn rem_assign(&mut self, rhs: Self) {
        if !NDArray::shape_equal(self, &rhs) {
            panic!("cannot rem assign");
        }
        let buffored_output = rem_two_vectors(&self._data.borrow(), &rhs._data.borrow());
        *self._data.borrow_mut() = buffored_output;
    }
}

impl<T: NDArrayBounds> Add for NDArray<T> {
    type Output = Option<Self>;

    fn add(self, rhs: Self) -> Self::Output {
        if !NDArray::shape_equal(&self, &rhs) {
            return None;
        }
        let buffored_output = add_two_vectors(&self._data.borrow(), &rhs._data.borrow());
        let new_array = NDArray {
            _data: Rc::new(RefCell::new(buffored_output)),
            _shape: self._shape.clone(),
            _strides: self._strides.clone(),
        };
        Some(new_array)
    }
}

impl<T: NDArrayBounds> Sub for NDArray<T> {
    type Output = Option<Self>;

    fn sub(self, rhs: Self) -> Self::Output {
        if !NDArray::shape_equal(&self, &rhs) {
            return None;
        }
        let buffored_output = substract_two_vectors(&self._data.borrow(), &rhs._data.borrow());
        let new_array = NDArray {
            _data: Rc::new(RefCell::new(buffored_output)),
            _shape: self._shape.clone(),
            _strides: self._strides.clone(),
        };
        Some(new_array)
    }
}

impl<T: NDArrayBounds> Mul for NDArray<T> {
    type Output = Option<Self>;

    fn mul(self, rhs: Self) -> Self::Output {
        if !NDArray::shape_equal(&self, &rhs) {
            return None;
        }
        let buffored_output = mul_two_vectors(&self._data.borrow(), &rhs._data.borrow());
        let new_array = NDArray {
            _data: Rc::new(RefCell::new(buffored_output)),
            _shape: self._shape.clone(),
            _strides: self._strides.clone(),
        };
        Some(new_array)
    }
}

impl<T: NDArrayBounds> Div for NDArray<T> {
    type Output = Option<Self>;

    fn div(self, rhs: Self) -> Self::Output {
        if !NDArray::shape_equal(&self, &rhs) {
            return None;
        }
        let buffored_output = div_two_vectors(&self._data.borrow(), &rhs._data.borrow());
        let new_array = NDArray {
            _data: Rc::new(RefCell::new(buffored_output)),
            _shape: self._shape.clone(),
            _strides: self._strides.clone(),
        };
        Some(new_array)
    }
}

impl<T: NDArrayBounds> Rem for NDArray<T> {
    type Output = Option<Self>;

    fn rem(self, rhs: Self) -> Self::Output {
        if !NDArray::shape_equal(&self, &rhs) {
            return None;
        }
        let buffored_output = rem_two_vectors(&self._data.borrow(), &rhs._data.borrow());
        let new_array = NDArray {
            _data: Rc::new(RefCell::new(buffored_output)),
            _shape: self._shape.clone(),
            _strides: self._strides.clone(),
        };
        Some(new_array)
    }
}

fn add_two_vectors<T: NDArrayBounds>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    a.iter().zip(b.iter())
        .map(|(a, b)| *a + *b)
        .collect()
}

fn substract_two_vectors<T: NDArrayBounds>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    a.iter().zip(b.iter())
        .map(|(a, b)| *a - *b)
        .collect()
}

fn mul_two_vectors<T: NDArrayBounds>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    a.iter().zip(b.iter())
        .map(|(a, b)| *a * *b)
        .collect()
}

fn div_two_vectors<T: NDArrayBounds>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    a.iter().zip(b.iter())
        .map(|(a, b)| *a / *b)
        .collect()
}

fn rem_two_vectors<T: NDArrayBounds>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    a.iter().zip(b.iter())
        .map(|(a, b)| *a % *b)
        .collect()
}