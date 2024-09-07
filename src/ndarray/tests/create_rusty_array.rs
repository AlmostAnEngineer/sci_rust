#[cfg(test)]
mod tests_create_rusty_array {
    use crate::ndarray::NDArray;
    #[test]
    fn create_rusty_array_i8(){
        let vec_usize: Vec<usize> = vec!(1,2,3);
        let initial_val: u8 = 10;
        let _array_r: NDArray<u8> = NDArray::full(vec_usize, initial_val);
    }
}
