#[cfg(test)]
mod integration_ndarray_adding {
    use sci_rust::ndarray::NDArray;
    #[test]
    fn adding_usize_should_be_equal() {
        let vec_size = vec![1,2,3];
        let a = NDArray::<usize>::ones(&vec_size);
        let b = NDArray::<usize>::zeros(&vec_size);

        let c = (&a + &b).unwrap();
        assert!(&a == &c);
    }
}
