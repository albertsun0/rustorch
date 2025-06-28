use crate::numrust::{
    tensor::Tensor, tensor_1::Tensor1, tensor_2::Tensor2, tensor_base::TensorBase,
};

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
