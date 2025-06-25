mod numrust;
use crate::numrust::{tensor::Tensor, tensor_base::TensorBase};

fn main() {
    println!("Hello, world!");

    let mut t = Tensor::Tensor1::<i32>(10);
    let x = t.get(&[0]);
    println!("{}", t.to_string());
}
