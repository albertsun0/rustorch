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

    pub fn scalar_mul(&self, rhs: T) -> Tensor2<T> {
        let data = self.data.iter().map(|&x| x * rhs).collect();
        Tensor2::from_vec(self.shape.clone(), data)
    }

    pub fn scalar_add(&self, rhs: T) -> Tensor2<T> {
        let data = self.data.iter().map(|&x| x + rhs).collect();
        Tensor2::from_vec(self.shape.clone(), data)
    }

    pub fn tensor_add(&self, other: &Tensor2<T>) -> Tensor2<T> {
        assert!(self.shape == other.shape, "shape mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor2::from_vec(self.shape.clone(), data)
    }

    pub fn norm(&self) -> T {
        let mut sum = T::default();
        for x in &self.data {
            sum = sum + x.clone() * x.clone();
        }
        sum.sqrt()
    }

    pub fn get_row(&self, i: usize) -> Tensor2<T> {
        let mut data = vec![T::default(); self.shape[1]];
        for j in 0..self.shape[1] {
            data[j] = self.data[i * self.shape[1] + j].clone();
        }
        Tensor2::from_vec(vec![1, self.shape[1]], data)
    }

    pub fn get_col(&self, i: usize) -> Tensor2<T> {
        let mut data = vec![T::default(); self.shape[0]];
        for j in 0..self.shape[0] {
            data[j] = self.data[j * self.shape[1] + i].clone();
        }
        Tensor2::from_vec(vec![self.shape[0], 1], data)
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

    // find x st. Ax = b, when A is lower triangular, diagonal 1
    pub fn forward_substitution(&self, b: &Tensor2<T>) -> Tensor2<T> {
        assert!(self.rows() == b.rows(), "matrix dimensions don't match");
        let mut result = b.copy();

        for i in 1..self.rows() {
            for j in 0..i {
                let multiplier = self.get(&[i, j]).clone();
                for k in 0..b.cols() {
                    result.set(
                        &[i, k],
                        result.get(&[i, k]).clone() - multiplier * result.get(&[j, k]).clone(),
                    );
                }
            }
        }

        result
    }

    // upper triangular, diagonal any
    pub fn backward_substitution(&self, b: &Tensor2<T>) -> Tensor2<T> {
        assert!(self.rows() == b.rows(), "matrix dimensions don't match");
        let mut result = b.copy();

        for i in (0..self.rows() - 1).rev() {
            for j in i + 1..self.rows() {
                let multiplier = self.get(&[i, j]).clone();
                for k in 0..b.cols() {
                    result.set(
                        &[i, k],
                        result.get(&[i, k]).clone() - multiplier * result.get(&[j, k]).clone(),
                    );
                }
            }

            // divide by diagonal
            let d = self.get(&[i, i]).clone();
            for k in 0..b.cols() {
                result.set(&[i, k], result.get(&[i, k]).clone() / d.clone());
            }
        }

        result
    }

    pub fn system_solve(&self, b: &Tensor2<T>) -> Tensor2<T> {
        /*
           solve Ax = b
           > decompose A st. PA = LU, or A = P^-1 LU
           > LUx = Pb
           > x = U^-1 L^-1 Pb
                > y = L^-1 Pb > Ly = Pb > forward substitution
        */
        let (p, l, u, _) = self.plu();

        let b_permute = p.mult(&b);
        let y = l.forward_substitution(&b_permute);
        u.backward_substitution(&y)
    }

    /*
       Inverse
       A = PLU, want X st. AX = I
        > PLUX = I
        > UX = L^-1 P^-1
        > solve using forward and backward substitution
    */
    pub fn inverse(&self) -> Tensor2<T> {
        let (p, l, u, _) = self.plu();

        let y = l.forward_substitution(&p);
        u.backward_substitution(&y)
    }

    /*
        QR decomposition

        Q is a orthogonal matrix (QQ^T = I)
        R is an upper triangular matrix

        On each update

        x x x    x x x
        x x x -> 0 x x
        x x x    0 x x

        Find householder transformation (mirror) H that maps first column to multiple of first unit basis vector

        R = HHHHH...A
        Q = IHHHHH... (since H^-1 = H)

        REF: https://www.cs.utexas.edu/~flame/laff/alaff/chapter03-householder-transformation.html
    */
    pub fn qr_decompose(&self) -> (Tensor2<T>, Tensor2<T>) {
        let mut R = self.copy();
        let rows = self.rows();
        let cols = self.cols();

        let mut Q: Tensor2<T> = Tensor2::identity(rows);

        for col in 0..rows.min(cols) {
            // compute u on a submatrix/subcolumn for efficiency
            let mut x_sub = Tensor2::<T>::new(rows - col, 1);
            for i in col..rows {
                x_sub.set(&[i - col, 0], R.get(&[i, col]).clone());
            }

            let beta = x_sub.norm();
            if beta == T::default() {
                continue;
            }

            let sign = if x_sub.get(&[0, 0]).clone() >= T::default() {
                T::one()
            } else {
                -T::one()
            };

            x_sub.set(&[0, 0], x_sub.get(&[0, 0]).clone() + sign * beta);

            let norm_v = x_sub.norm();

            if norm_v == T::default() {
                continue;
            }

            let u_sub = x_sub * (T::one() / norm_v);
            // turn u_sub into tall matrix
            let mut u = Tensor2::<T>::new(rows, 1);
            for i in 0..(rows - col) {
                u.set(&[i + col, 0], u_sub.get(&[i, 0]).clone());
            }

            let H = Tensor2::identity(rows) - u.mult(&u.transpose()) * T::from(2.0);

            R = H.mult(&R);
            Q = Q.mult(&H);
        }

        (Q, R)
    }
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
        let temp = self.copy();
        temp.scalar_add(rhs)
    }
}
// scalar subtraction
impl<T: TensorNumber> ops::Sub<T> for Tensor2<T> {
    type Output = Tensor2<T>;

    fn sub(self, rhs: T) -> Self::Output {
        self.scalar_add(rhs * -T::one())
    }
}
impl<'a, T: TensorNumber> ops::Sub<T> for &'a Tensor2<T> {
    type Output = Tensor2<T>;

    fn sub(self, rhs: T) -> Self::Output {
        let temp = self.copy();
        temp.scalar_add(rhs * -T::one())
    }
}

// equals
impl<T: TensorNumber> PartialEq for Tensor2<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

// tensor 2 addition
impl<T: TensorNumber> ops::Add<Tensor2<T>> for Tensor2<T> {
    type Output = Tensor2<T>;

    fn add(self, rhs: Tensor2<T>) -> Self::Output {
        self.tensor_add(&rhs)
    }
}
impl<'a, T: TensorNumber> ops::Add<Tensor2<T>> for &'a Tensor2<T> {
    type Output = Tensor2<T>;

    fn add(self, rhs: Tensor2<T>) -> Self::Output {
        let temp = self.copy();
        temp.tensor_add(&rhs)
    }
}

impl<T: TensorNumber> ops::Sub<Tensor2<T>> for Tensor2<T> {
    type Output = Tensor2<T>;

    fn sub(self, rhs: Tensor2<T>) -> Self::Output {
        self.tensor_add(&(rhs * -T::one()))
    }
}
impl<'a, T: TensorNumber> ops::Sub<Tensor2<T>> for &'a Tensor2<T> {
    type Output = Tensor2<T>;

    fn sub(self, rhs: Tensor2<T>) -> Self::Output {
        let temp = self.copy();
        temp.tensor_add(&(rhs * -T::one()))
    }
}

impl<T: TensorNumber> ops::Mul<&Tensor2<T>> for Tensor2<T> {
    type Output = Tensor2<T>;

    fn mul(self, rhs: &Tensor2<T>) -> Self::Output {
        self.mult(&rhs)
    }
}
impl<'a, T: TensorNumber> ops::Mul<&Tensor2<T>> for &'a Tensor2<T> {
    type Output = Tensor2<T>;

    fn mul(self, rhs: &Tensor2<T>) -> Self::Output {
        let temp = self.copy();
        temp.mult(&rhs)
    }
}
