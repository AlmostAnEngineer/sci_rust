use num::Num;

#[derive(Debug)]
pub struct RustyArray<T>{
    _data: Vec<T>,
    _shape: Vec<usize>,
    _strides: Vec<usize>
}

impl<T: Num + Copy> RustyArray<T> {
    pub fn new(size: Vec<usize>, initial_value: T) -> Self {
        // creates new dynamic array 
        // T must be numeric
        let vec_size: usize = Self::compute_array_size(&size);
        let strides: Vec<usize> = Self::calculate_strides(&size);
        let data = vec![initial_value; vec_size];
        return RustyArray{_data: data, _shape: size, _strides: strides};
    }

    fn compute_array_size(size: &Vec<usize>) -> usize {
        //full size needed to allocate array is product of all items
        let full_size: usize = size.iter().product();
        return full_size;
    }

    fn calculate_strides(shape: &Vec<usize>) -> Vec<usize> {
        let mut strides: Vec<usize> = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1]
        }
        return strides;
    }

    pub fn get_data(&self) -> &Vec<T> {
        // return non-mutable object inside array
        return &self._data;
    }

    pub fn clear(&mut self) {
        self._data.clear();
        self._strides.clear();
        self._shape.clear();
    }

}

