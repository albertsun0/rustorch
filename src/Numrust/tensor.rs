use crate::numrust::tensor_1::Tensor1;
pub struct Tensor;

impl Tensor {
    pub fn Tensor1<T: Clone + Default>(len: usize) -> Tensor1<T> {
        Tensor1 {
            data: vec![T::default(); len],
        }
    }
}
