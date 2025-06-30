use std::fmt;

use std::ops::{Add, Div, Mul, Sub, Neg};

/// Trait for types usable in tensors
pub trait TensorNumber:
    Copy               // cheap to copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self> 
    + Default          // for zero value
    + PartialEq        // equality comparisons
    + std::fmt::Debug  // for logging
    + std::cmp::PartialOrd
    + std::ops::Neg
    + std::ops::MulAssign
{
    fn rand() -> Self;
    fn one() -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn from(x: f64) -> Self;
}


macro_rules! impl_tensor_number {
    ($($t:ty),*) => {
        $(
            impl TensorNumber for $t {
                fn from(x: f64) -> Self { x as $t }
                fn one() -> Self { 1 as $t }
                // TODO: Implement pseudorandom generator for TensorNumber
                fn rand() -> Self { 1 as $t }
                fn abs(self) -> Self { <$t>::abs(self) }
                 fn sqrt(self) -> Self {
                    <$t>::sqrt(self)
                }
            }
        )*
    };
}

impl_tensor_number!(f32, f64);

pub trait TensorBase<T: TensorNumber> {
    fn shape(&self) -> Vec<usize>;
    fn data(&self) -> &Vec<T>;
    fn data_mut(&mut self) -> &mut Vec<T>;
    fn flatten_index(&self, indices: &[usize]) -> usize;

    fn to_string(&self) -> String {
        format!(
            "TensorBase {{ shape: {:?}, data: {:?} }}",
            self.shape(),
            self.data()
        )
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
        let idx: usize = self.flatten_index(indices);
        self.data_mut()[idx] = val;
    }
}
