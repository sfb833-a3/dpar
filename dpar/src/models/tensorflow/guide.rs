use enum_map::EnumMap;
use tensorflow::Tensor;

use features::Layer;
use system::{ParserState, TransitionSystem};

use guide::{BatchGuide, Guide};

use super::{TensorWrap, TensorflowModel};

impl<T> Guide for TensorflowModel<T>
where
    T: TransitionSystem,
{
    type T = T::T;

    fn best_transition(&mut self, state: &ParserState) -> Self::T {
        self.best_transitions(&[state]).remove(0)
    }
}

impl<T> BatchGuide for TensorflowModel<T>
where
    T: TransitionSystem,
{
    type T = T::T;

    fn best_transitions(&mut self, states: &[&ParserState]) -> Vec<Self::T> {
        if states.is_empty() {
            return Vec::new();
        }

        // Allocate batch tensors.
        let mut input_tensors = EnumMap::new();
        for (layer, size) in self.vectorizer().layer_sizes() {
            input_tensors[layer] = Tensor::new(&[states.len() as u64, size as u64]).into();
        }

        // Fill tensors.
        for (idx, state) in states.iter().enumerate() {
            self.vectorizer().realize_into(
                state,
                &mut batch_to_instance_slices(&mut input_tensors, idx),
            );
        }

        let logits = self.predict_logits(&input_tensors);

        // Get the best transition for each parser state.
        let n_labels = logits.dims()[1] as usize;
        states
            .iter()
            .enumerate()
            .map(|(idx, state)| {
                let offset = idx * n_labels;
                self.logits_best_transition(state, &logits[offset..offset + n_labels])
            }).collect()
    }
}

/// Extract for each layer the slice corresponding to the `idx`-th
/// instance from the batch.
fn batch_to_instance_slices<'a>(
    batch_tensors: &'a mut EnumMap<Layer, TensorWrap>,
    idx: usize,
) -> EnumMap<Layer, &'a mut [i32]> {
    let mut slices = EnumMap::new();

    for (layer, tensor) in batch_tensors {
        let layer_size = tensor.dims()[1] as usize;
        let offset = idx * layer_size;
        slices[layer] = &mut tensor[offset..offset + layer_size];
    }

    slices
}
