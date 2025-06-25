use std::fmt;

use std::ops::{Add, Div, Mul, Sub};

/// Trait for types usable in tensors without external crates
pub trait TensorNumber:
    Copy               // cheap to copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Default          // for zero value
    + PartialEq        // equality comparisons
    + std::fmt::Debug  // for logging
{}

impl<T> TensorNumber for T where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Default
        + PartialEq
        + std::fmt::Debug
{
}

pub trait TensorBase<T: TensorNumber> {
    fn shape(&self) -> Vec<usize>;
    fn data(&self) -> &Vec<T>;
    fn data_mut(&mut self) -> &mut Vec<T>;
    fn flatten_index(&self, indices: &[usize]) -> usize;

    fn to_string(&self) -> String {
        format!("TensorBase {{ data: {:?} }}", self.data())
    }

    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Default method: get a reference to an element
    fn get(&self, indices: &[usize]) -> &T {
        &self.data()[self.flatten_index(indices)]
    }

    /// Default method: set a value
    fn set(&mut self, indices: &[usize], val: T) {
        let idx = self.flatten_index(indices);
        self.data_mut()[idx] = val;
    }
}
