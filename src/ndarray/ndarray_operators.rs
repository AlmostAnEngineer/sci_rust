use std::ops::{Add, Sub, Mul, Div};
use num::{FromPrimitive, Num};
use crate::ndarray::NDArray;

impl<T: Num + FromPrimitive + Copy> Add for NDArray<T>
{
    type Output = Option<Self>;
    fn add(self, rhs: Self) -> Self::Output {
        if self._shape != rhs._shape {
            return None;
        }
        let buffored_output: Vec<T> = self._data.iter().zip(rhs._data.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        let new_array = NDArray{_data: buffored_output, _shape: self._shape.clone(), _strides: self._strides.clone()};
        return Some(new_array);
    }
}

impl<T: Num + FromPrimitive + Copy> Sub for NDArray<T>
{
    type Output = Option<Self>;
    fn sub(self, rhs: Self) -> Self::Output {
        if self._shape != rhs._shape {
            return None;
        }
        let buffored_output: Vec<T> = self._data.iter().zip(rhs._data.iter())
            .map(|(a, b)| *a - *b)
            .collect();
        let new_array = NDArray{_data: buffored_output, _shape: self._shape.clone(), _strides: self._strides.clone()};
        return Some(new_array);
    }
}

impl<T: Num + Copy> Mul for NDArray<T> {
    type Output = Option<Self>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self._shape != rhs._shape {
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

impl<T: Num + Copy> Div for NDArray<T> {
    type Output = Option<Self>;

    fn div(self, rhs: Self) -> Self::Output {
        if self._shape != rhs._shape {
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