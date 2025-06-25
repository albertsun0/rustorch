use crate::numrust::tensor_base::{TensorBase, TensorNumber};

pub struct Tensor1<T> {
    pub data: Vec<T>,
}

impl<T: TensorNumber + Clone> TensorBase<T> for Tensor1<T> {
    fn shape(&self) -> Vec<usize> {
        vec![self.data.len()]
    }
    fn data(&self) -> &Vec<T> {
        &self.data
    }
    fn data_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }
    fn flatten_index(&self, indices: &[usize]) -> usize {
        indices[0]
    }
}
