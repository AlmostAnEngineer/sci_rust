#[cfg(test)]
mod tests_create_ndarrays {
    use crate::ndarray::NDArray;

    #[test]
    fn create_ndarray_random(){
        let vec_usize: Vec<usize> = vec!(1,2,3);
        let array_a: NDArray<u8> = NDArray::random(&vec_usize);
        let array_b: NDArray<u8> = NDArray::random(&vec_usize);
        assert_ne!(array_a._data, array_b._data, "The two randomly generated arrays should not be identical.");
    }
}
