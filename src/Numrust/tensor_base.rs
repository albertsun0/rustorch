use std::fmt;

use std::ops::{Add, Div, Mul, Sub};

/// Trait for types usable in tensors
pub trait TensorNumber:
    Copy               // cheap to copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Default          // for zero value
    + PartialEq        // equality comparisons
    + std::fmt::Debug  // for logging
{
    fn rand() -> Self;
    fn one() -> Self;
}

// impl<T> TensorNumber for T
// where
//     T: Copy
//         + Add<Output = T>
//         + Sub<Output = T>
//         + Mul<Output = T>
//         + Div<Output = T>
//         + Default
//         + PartialEq
//         + std::fmt::Debug,
// {
//     // TODO: Implement pseudorandom generator for TensorNumber
//     fn rand() -> Self {
//         Self::default()
//     }
// }

macro_rules! impl_tensor_number {
    ($($t:ty),*) => {
        $(
            impl TensorNumber for $t {
                fn one() -> Self { 1 as $t }
                // TODO: Implement pseudorandom generator for TensorNumber
                fn rand() -> Self { 1 as $t }
            }
        )*
    };
}

impl_tensor_number!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, usize, f32, f64);

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
        let idx = self.flatten_index(indices);
        self.data_mut()[idx] = val;
    }
}
