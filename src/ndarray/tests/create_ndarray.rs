#[cfg(test)]
mod tests_create_ndarrays {
    use crate::ndarray::NDArray;
    #[test]
    fn create_ndarray_full(){
        let vec_usize: Vec<usize> = vec!(1,2,3);
        let initial_val: u8 = 10;
        let _array: NDArray<u8> = NDArray::full(&vec_usize, initial_val);
        assert!(_array._data.iter().all(|a| *a == 10));
        assert_eq!(_array._shape, vec![1, 2, 3])
    }

    #[test]
    fn create_ndarray_ones(){
        let vec_usize: Vec<usize> = vec!(1,2,3);
        let _array: NDArray<u8> = NDArray::ones(&vec_usize);
        assert!(_array._data.iter().all(|a| *a == 1));
        assert_eq!(_array._shape, vec![1, 2, 3])
    }

    #[test]
    fn create_ndarray_zeros(){
        let vec_usize: Vec<usize> = vec!(1,2,3);
        let _array: NDArray<u8> = NDArray::zeros(&vec_usize);
        assert!(_array._data.iter().all(|a| *a == 0), "Some elements are not zeros");
        assert_eq!(_array._shape, vec![1, 2, 3], "shapes are different")
    }

    #[test]
    fn create_ndarray_random(){
        let vec_usize: Vec<usize> = vec!(1,2,3);
        let array_a: NDArray<u8> = NDArray::random(&vec_usize);
        let array_b: NDArray<u8> = NDArray::random(&vec_usize);
        assert_ne!(array_a._data, array_b._data, "The two randomly generated arrays should not be identical.");
    }
}
