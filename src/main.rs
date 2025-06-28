mod numrust;

use crate::numrust::{
    tensor::Tensor, tensor_1::Tensor1, tensor_2::Tensor2, tensor_base::TensorBase,
};

fn main() {
    println!("Hello, world!");

    let mut t = Tensor::Tensor1::<i32>(10);

    t.set(&[0], 1);

    let mut t2 = &t * 2;

    let mut t3 = t2.dot(&t2);

    let mut t4 = Tensor1::from_vec(t3.data.clone());

    let x = t.get(&[0]);
    println!("{}", t.to_string());
    println!("{}", t2.to_string());
    println!("{}", t3.to_string());
    println!("{:?}", t.shape());

    let t5 = Tensor2::from_vec(vec![2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);

    println!("{}", t5.to_string());

    let mut t6 = t5.transpose();

    println!("{}", t6.to_string());

    let mut t7 = t5.mult(&t6);

    println!("{}", t7.to_string());

    let mut identity = Tensor2::<f64>::identity(10) * 2.0;

    println!("{}", identity.to_string());

    let t6 = Tensor2::from_vec(
        vec![3, 3],
        vec![6.0, 18.0, 3.0, 2.0, 12.0, 1.0, 4.0, 15.0, 3.0],
    );

    let t7 = Tensor2::from_vec(
        vec![3, 3],
        vec![2.0, -2.0, 1.0, 1.0, 3.0, -2.0, 3.0, -1.0, -1.0],
    );

    let mut b: Tensor2<f64> = Tensor2::from_vec(vec![3, 1], vec![-3.0, 1.0, 2.0]);

    //solve t7x = b
    print!("{}", t7.system_solve(&b).to_string());

    let t7_inv = t7.inverse();
    print!("{}", t7_inv.to_string());

    print!("{}", t7.mult(&t7_inv).to_string());
}

// -7, -10, -21
