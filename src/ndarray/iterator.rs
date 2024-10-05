pub struct NDArrayIterator {
    shape: Vec<usize>,
    current: Vec<usize>,
    finished: bool,
}

impl NDArrayIterator {
    pub fn new(shape: Vec<usize>) -> Self {
        let current = vec![0; shape.len()];
        NDArrayIterator {
            shape,
            current,
            finished: false,
        }
    }

    fn increment_indices(&mut self) {
        for i in (0..self.shape.len()).rev() {
            if self.current[i] + 1 < self.shape[i] {
                self.current[i] += 1;
                return;
            } else {
                self.current[i] = 0;
            }
        }
        self.finished = true;

    }
}
impl Iterator for NDArrayIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let result = self.current.clone();
        self.increment_indices();
        Some(result)
    }
}