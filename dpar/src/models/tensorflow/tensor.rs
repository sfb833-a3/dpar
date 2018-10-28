use std::ops::{Deref, DerefMut};

use enum_map::EnumMap;

use features::Layer;
use tensorflow::{Tensor, TensorType};

/// Ad-hoc trait for copying a subset of batches.
pub trait CopyBatches {
    fn copy_batches(&self, n_batches: u64) -> Self;
}

impl<T> CopyBatches for Tensor<T>
where
    T: Copy + TensorType,
{
    fn copy_batches(&self, n_batches: u64) -> Self {
        assert!(n_batches <= self.dims()[0]);

        let mut new_shape = self.dims().to_owned();
        new_shape[0] = n_batches;
        let mut copy = Tensor::new(&new_shape);

        copy.copy_from_slice(&self[..new_shape.iter().cloned().product::<u64>() as usize]);

        copy
    }
}

impl<T> CopyBatches for TensorWrap<T>
where
    T: Copy + TensorType,
{
    fn copy_batches(&self, n_batches: u64) -> Self {
        TensorWrap(self.0.copy_batches(n_batches))
    }
}

/// Layer-wise batch tensors.
///
/// Instances of this type store the per-layer inputs for a batch.
pub struct LayerTensors(pub EnumMap<Layer, TensorWrap<i32>>);

impl LayerTensors {
    /// Extract for each layer the slice corresponding to the `idx`-th
    /// instance from the batch.
    pub fn to_instance_slices(&mut self, idx: usize) -> EnumMap<Layer, &mut [i32]> {
        let mut slices = EnumMap::new();

        for (layer, tensor) in self.iter_mut() {
            let layer_size = tensor.dims()[1] as usize;
            let offset = idx * layer_size;
            slices[layer] = &mut tensor[offset..offset + layer_size];
        }

        slices
    }
}

impl Deref for LayerTensors {
    type Target = EnumMap<Layer, TensorWrap<i32>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LayerTensors {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Simple wrapper for `Tensor` that implements the `Default`
/// trait.
pub struct TensorWrap<T>(pub Tensor<T>)
where
    T: TensorType;

impl<T> Default for TensorWrap<T>
where
    T: TensorType,
{
    fn default() -> Self {
        TensorWrap(Tensor::new(&[]))
    }
}

impl<T> From<Tensor<T>> for TensorWrap<T>
where
    T: TensorType,
{
    fn from(tensor: Tensor<T>) -> Self {
        TensorWrap(tensor)
    }
}

impl<T> Deref for TensorWrap<T>
where
    T: TensorType,
{
    type Target = Tensor<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for TensorWrap<T>
where
    T: TensorType,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
