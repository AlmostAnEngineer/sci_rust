use std::ops::{Add, Sub, Mul, Div, Rem};
use std::cmp::PartialEq;
use crate::ndarray::ndarray_traits::NDArrayBounds;
use crate::ndarray::NDArray;

impl<T: NDArrayBounds> Add for NDArray<T>
{
    type Output = Option<Self>;
    fn add(self, rhs: Self) -> Self::Output {
        if !NDArray::shape_equal(&self, &rhs) {
            return None;
        }
        let buffored_output: Vec<T> = self._data.iter().zip(rhs._data.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        let new_array = NDArray{_data: buffored_output, _shape: self._shape.clone(), _strides: self._strides.clone()};
        Some(new_array)
    }
}

impl<T: NDArrayBounds> Sub for NDArray<T>
{
    type Output = Option<Self>;
    fn sub(self, rhs: Self) -> Self::Output {
        if !NDArray::shape_equal(&self, &rhs) {
            return None;
        }
        let buffored_output: Vec<T> = self._data.iter().zip(rhs._data.iter())
            .map(|(a, b)| *a - *b)
            .collect();
        let new_array = NDArray{_data: buffored_output, _shape: self._shape.clone(), _strides: self._strides.clone()};
        Some(new_array)
    }
}

impl<T: NDArrayBounds> Mul for NDArray<T> {
    type Output = Option<Self>;

    fn mul(self, rhs: Self) -> Self::Output {
        if !NDArray::shape_equal(&self, &rhs) {
            return None;
        }

        let result_data = self._data.iter().zip(rhs._data.iter())
            .map(|(a, b)| *a * *b)
            .collect();

        Some(NDArray {
            _shape: self._shape.clone(),
            _strides: self._strides.clone(),
            _data: result_data,
        })
    }
}

impl<T: NDArrayBounds> Div for NDArray<T> {
    type Output = Option<Self>;

    fn div(self, rhs: Self) -> Self::Output {
        if !NDArray::shape_equal(&self, &rhs) {
            return None;
        }

        let result_data = self._data.iter().zip(rhs._data.iter())
            .map(|(a, b)| *a / *b)
            .collect();

        Some(NDArray {
            _shape: self._shape.clone(),
            _strides: self._strides.clone(),
            _data: result_data,
        })
    }
}

impl<T: NDArrayBounds> Rem for NDArray<T> {
    type Output = Option<Self>;
    fn rem(self, rhs: Self) -> Self::Output {
        if !NDArray::shape_equal(&self, &rhs) {
            return None;
        }
        let result_data = self._data.iter().zip(rhs._data.iter())
            .map(|(a, b)| *a % *b)
            .collect();

        Some(NDArray {
            _shape: self._shape.clone(),
            _strides: self._strides.clone(),
            _data: result_data,
        })
    }
}

impl<T: NDArrayBounds> PartialEq for NDArray<T> {
    fn eq(&self, other: &NDArray<T>) -> bool {
        if !NDArray::shape_equal(self, other) {
            return false;
        }
        let result_data = self._data.iter().eq(other._data.iter());
        result_data
    }
}
