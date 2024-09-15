#[cfg(test)]
mod tests_add_ndarrays {
    use crate::ndarray::NDArray;
    #[test]
    fn adding_usize_should_be_equal() {
        let vec_size = vec![1,2,3];
        let a: NDArray<usize> = NDArray::ones(&vec_size);
        let b: NDArray<usize> = NDArray::zeros(&vec_size);

        let add_result = a.clone() + b.clone();
        let add = add_result.expect("ndim mismatch");
        assert_eq!(add._data, a._data, "should be same");
    }

    #[test]
    fn adding_usize_should_not_be_equal() {
        let vec_size = vec![1,2,3];
        let a: NDArray<usize> = NDArray::ones(&vec_size);
        let b: NDArray<usize> = NDArray::zeros(&vec_size);

        let add_result = a.clone() + b.clone();
        let add = add_result.expect("ndim mismatch");
        assert_ne!(add._data, b._data, "should not be same");
    }
}
