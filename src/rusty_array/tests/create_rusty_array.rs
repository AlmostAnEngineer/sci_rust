#[cfg(test)]
mod tests_create_rusty_array {
use crate::rusty_array::rusty_array::RustyArray;
    #[test]
    fn create_rusty_array_i8(){
        let vec_usize: Vec<usize> = vec!(1,2,3);
        let initial_val: u8 = 10;
        let array_r: RustyArray<u8> = RustyArray::new(vec_usize, initial_val);
    }
}
