use crate::ndarray::NDArray;
use rand::distributions::{Distribution, Uniform};
use crate::ndarray::traits::NDArrayBounds;
use std::rc::Rc;
use std::cell::RefCell;

impl<T> NDArray<T>
where
    T: NDArrayBounds,
    Uniform<T>: Distribution<T>, {
    pub fn ones(size: &Vec<usize>) -> Self {
        Self::create_constant_values_vec(size.clone(), T::from_u32(1).unwrap())
    }

    pub fn zeros(size: &Vec<usize>) -> Self {
        Self::create_constant_values_vec(size.clone(), T::from_u32(0).unwrap())
    }

    pub fn full(size: &Vec<usize>, fill_value: T) -> Self {
        Self::create_constant_values_vec(size.clone(), fill_value)
    }

    pub fn random(size: &Vec<usize>) -> Self {
        Self::create_random_values_vec(size.clone())
    }

    pub fn shape(&self) -> Vec<usize> {
        self._shape.borrow().clone()
    }

    pub fn clear(&mut self) {
        self._data.borrow_mut().clear();
        self._shape.borrow_mut().clear();
        self._strides.borrow_mut().clear();
    }

    pub fn get(&self, cords: &[usize]) -> Result<T, String> {
        if cords.len() != self._shape.borrow().len() {
            return Err("Cords len mismatch ndim!".to_string());
        }
        let idx = self.index(cords);
        Ok(self._data.borrow()[idx].clone())
    }

    pub fn ndim(&self) -> usize {
        self._shape.borrow().len()
    }

    fn compute_array_size(size: &Vec<usize>) -> usize {
        size.iter().product()
    }

    fn calculate_strides(shape: &Vec<usize>) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    fn create_random_values_vec(size: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        let vec_size = Self::compute_array_size(&size);
        let strides = Self::calculate_strides(&size);
        let mut data = Vec::with_capacity(vec_size);

        let min = T::min_value();
        let max = T::max_value();
        let between = Uniform::from(min..max);

        for _ in 0..vec_size {
            let random_value: T = between.sample(&mut rng);
            data.push(random_value)
        }

        NDArray {
            _data: Rc::new(RefCell::new(data)),
            _shape: Rc::new(RefCell::new(size)),
            _strides: Rc::new(RefCell::new(strides)),
        }
    }

    fn create_constant_values_vec(size: Vec<usize>, initial_value: T) -> Self {
        let vec_size = Self::compute_array_size(&size);
        let strides = Self::calculate_strides(&size);
        let data = vec![initial_value; vec_size];

        NDArray {
            _data: Rc::new(RefCell::new(data)),
            _shape: Rc::new(RefCell::new(size)),
            _strides: Rc::new(RefCell::new(strides)),
        }
    }

    pub fn index(&self, indices: &[usize]) -> usize {
        indices
            .iter()
            .zip(self._strides.borrow().iter())
            .map(|(i, stride)| i * stride)
            .sum()
    }

    #[cfg(test)]
    pub fn debug_create_raw(_data: Vec<T>, _shape: Vec<usize>, _strides: Vec<usize>) -> Self {
        NDArray {
            _data: Rc::new(RefCell::new(_data)),
            _shape: Rc::new(RefCell::new(_shape)),
            _strides: Rc::new(RefCell::new(_strides)),
        }
    }
}
