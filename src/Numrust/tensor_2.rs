use crate::numrust::tensor_base::{TensorBase, TensorNumber};
use std::ops;

pub struct Tensor2<T: TensorNumber> {
    shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T: TensorNumber + Clone> TensorBase<T> for Tensor2<T> {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
    fn data(&self) -> &Vec<T> {
        &self.data
    }
    fn data_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }
    fn flatten_index(&self, indices: &[usize]) -> usize {
        assert!(indices.len() == 2);
        assert!(
            indices[0] * self.shape[1] + indices[1] < self.data.len(),
            "index out of bounds [{},{}] for shape {:?}",
            indices[0],
            indices[1],
            self.shape
        );
        indices[0] * self.cols() + indices[1]
    }

    fn to_string(&self) -> String {
        let mut str = String::new();

        str += &format!("Tensor2 {{ shape: {:?}, data: [\n", self.shape);

        for i in 0..self.shape[0] {
            str += "\t[";
            for j in 0..self.shape[1] {
                str += &format!(
                    "{:?}{}",
                    self.get(&[i, j]),
                    if j < self.shape[1] - 1 { ", " } else { "" }
                );
            }
            str += "]\n";
        }
        str += "]}";
        str
    }
}

impl<T: TensorNumber> Tensor2<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        let shape = vec![rows, cols];
        Tensor2::from_vec(shape, vec![T::default(); rows * cols])
    }

    pub fn from_vec(shape: Vec<usize>, data: Vec<T>) -> Self {
        assert!(shape.len() == 2, "shape length must be 2");
        assert!(
            shape[0] * shape[1] == data.len(),
            "shape must match data size"
        );
        Tensor2 {
            shape: shape,
            data: data,
        }
    }

    pub fn is_square(&self) -> bool {
        self.shape[0] == self.shape[1]
    }

    pub fn square(n: usize) -> Tensor2<T> {
        Tensor2::from_vec(vec![n, n], vec![T::default(); n * n])
    }

    pub fn identity(n: usize) -> Tensor2<T> {
        let mut data = vec![T::default(); n * n];
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        Tensor2::from_vec(vec![n, n], data)
    }

    pub fn copy(&self) -> Tensor2<T> {
        Tensor2 {
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
    }

    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    pub fn scalar_mul(self, rhs: T) -> Tensor2<T> {
        let data = self.data.iter().map(|&x| x * rhs).collect();
        Tensor2::from_vec(self.shape.clone(), data)
    }

    pub fn scalar_add(self, rhs: T) -> Tensor2<T> {
        let data = self.data.iter().map(|&x| x + rhs).collect();
        Tensor2::from_vec(self.shape.clone(), data)
    }

    pub fn transpose(&self) -> Tensor2<T> {
        let new_shape = vec![self.shape[1], self.shape[0]];
        let mut new_data = vec![T::default(); new_shape[0] * new_shape[1]];

        for i in 0..new_shape[0] {
            for j in 0..new_shape[1] {
                new_data[i * new_shape[1] + j] = self.data[j * new_shape[0] + i].clone();
            }
        }

        Tensor2::from_vec(new_shape, new_data)
    }

    // TODO: chunking and SIMD optimization
    pub fn mult(&self, other: &Tensor2<T>) -> Tensor2<T> {
        assert!(
            self.shape[1] == other.shape[0],
            "shape mismatch, {} != {}",
            self.shape[1],
            other.shape[0]
        );

        let new_shape = vec![self.shape[0], other.shape[1]];

        let mut result = Tensor2::<T>::new(new_shape[0], new_shape[1]);

        for i in 0..new_shape[0] {
            for k in 0..self.shape[1] {
                let a_il = self.get(&[i, k]);
                for j in 0..new_shape[1] {
                    let b_kj = other.get(&[k, j]);
                    result.set(
                        &[i, j],
                        result.get(&[i, j]).clone() + a_il.clone() * b_kj.clone(),
                    );
                }
            }
        }

        result
    }

    /*
       Prepare for LU decomposition by swapping rows
    */
    pub fn swap_rows(&mut self, mut i: usize, mut j: usize) {
        if i == j {
            return;
        }

        let cols = self.cols();

        if j < i {
            std::mem::swap(&mut i, &mut j);
        }

        let (a, b) = self.data.split_at_mut(j * cols);
        let row_i = &mut a[i * cols..(i + 1) * cols];
        let row_j = &mut b[..cols];

        row_i.swap_with_slice(row_j);
    }
    /*
       Calculate P, L, U st. PA = LU
    */
    pub fn plu(&self) -> (Tensor2<T>, Tensor2<T>, Tensor2<T>, usize) {
        let rows = self.rows();
        let mut L = Tensor2::square(self.shape[0]);
        let mut U = self.copy();
        let mut permutation = Tensor2::identity(rows);
        let mut swaps = 0;

        for i in 0..rows - 1 {
            // set pivot & swap
            let mut max_row = i;
            let mut max_value = U.get(&[i, i]).clone().abs();
            for j in i + 1..rows {
                let row_value = U.get(&[j, i]).clone().abs();
                if row_value > max_value {
                    max_row = j;
                    max_value = row_value;
                }
            }

            if max_row != i {
                swaps += 1;
                permutation.swap_rows(i, max_row);
                U.swap_rows(i, max_row);
                L.swap_rows(i, max_row);
            }

            let val = U.get(&[i, i]).clone();
            assert!(val != T::default(), "matrix is singular");

            // set column in L
            for j in i + 1..rows {
                L.set(&[j, i], U.get(&[j, i]).clone() / val.clone());
            }

            // set U
            for j in i + 1..rows {
                for k in i..self.cols() {
                    let prev = U.get(&[i, k]).clone();
                    let multiplier = L.get(&[j, i]).clone();
                    let cur = U.get(&[j, k]).clone();
                    U.set(&[j, k], cur - prev * multiplier);
                }
            }
        }
        // set diagonal of L
        for i in 0..rows {
            L.set(&[i, i], T::one());
        }

        (permutation, L, U, swaps)
    }

    pub fn det(&self) -> T {
        assert!(self.is_square(), "matrix is not square");

        let (_, _, u, swaps) = self.plu();
        let mut result = T::one();
        for i in 0..self.rows() {
            result *= u.get(&[i, i]).clone();
        }
        if swaps % 2 == 1 {
            return -result;
        }
        result
    }

    // forward substitution
    // backward substitution
    // inverse
    // eigenvalues
}

// scalar multiplication
impl<T: TensorNumber> ops::Mul<T> for Tensor2<T> {
    type Output = Tensor2<T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.scalar_mul(rhs)
    }
}
impl<'a, T: TensorNumber> ops::Mul<T> for &'a Tensor2<T> {
    type Output = Tensor2<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let data = self.data.iter().map(|&x| x * rhs).collect();
        Tensor2::from_vec(self.shape.clone(), data)
    }
}

// scalar addition
impl<T: TensorNumber> ops::Add<T> for Tensor2<T> {
    type Output = Tensor2<T>;

    fn add(self, rhs: T) -> Self::Output {
        self.scalar_add(rhs)
    }
}
impl<'a, T: TensorNumber> ops::Add<T> for &'a Tensor2<T> {
    type Output = Tensor2<T>;

    fn add(self, rhs: T) -> Self::Output {
        let data = self.data.iter().map(|&x| x + rhs).collect();
        Tensor2::from_vec(self.shape.clone(), data)
    }
}

impl<T: TensorNumber> PartialEq for Tensor2<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}
