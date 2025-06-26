use crate::numrust::tensor_1::Tensor1;
use crate::numrust::tensor_2::Tensor2;
use crate::numrust::tensor_base::TensorNumber;
pub struct Tensor;

impl Tensor {
    pub fn Tensor1<T: TensorNumber>(len: usize) -> Tensor1<T> {
        Tensor1::new(len)
    }

    pub fn Tensor2<T: TensorNumber>(rows: usize, cols: usize) -> Tensor2<T> {
        Tensor2::new(rows, cols)
    }
}
