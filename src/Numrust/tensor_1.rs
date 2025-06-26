use crate::numrust::tensor_base::{TensorBase, TensorNumber};
use std::ops;

pub struct Tensor1<T: TensorNumber> {
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
        assert!(indices.len() == 1);
        indices[0]
    }
}

impl<T: TensorNumber> Tensor1<T> {
    pub fn new(len: usize) -> Self {
        Tensor1::from_vec(vec![T::default(); len])
    }

    pub fn rand(len: usize) -> Self {
        Tensor1::from_vec(vec![T::rand(); len])
    }
    pub fn from_vec(data: Vec<T>) -> Self {
        Tensor1 { data }
    }

    pub fn dot(&self, other: &Tensor1<T>) -> Tensor1<T> {
        assert!(self.shape() == other.shape());

        let mut res = vec![T::default(); self.shape()[0]];
        for i in 0..self.shape()[0] {
            res[i] = self.get(&[i]).clone() * other.get(&[i]).clone();
        }
        Tensor1::from_vec(res)
    }
}

impl<T: TensorNumber> ops::Mul<T> for Tensor1<T> {
    type Output = Tensor1<T>;

    fn mul(self, other: T) -> Self::Output {
        let res = self.data.iter().map(|x| x.clone() * other).collect();
        Tensor1::from_vec(res)
    }
}

impl<'a, T: TensorNumber> ops::Mul<T> for &'a Tensor1<T> {
    type Output = Tensor1<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let data = self.data.iter().map(|&x| x * rhs).collect();
        Tensor1::from_vec(data)
    }
}
