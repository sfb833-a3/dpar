use enum_map::EnumMap;
use tensorflow::Tensor;

use system::{ParserState, TransitionSystem};

use guide::{BatchGuide, Guide};

use super::{LayerTensors, TensorflowModel};

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
        let mut input_tensors = LayerTensors(EnumMap::new());
        for (layer, size) in self.vectorizer().layer_sizes() {
            input_tensors[layer] = Tensor::new(&[states.len() as u64, size as u64]).into();
        }

        // Fill tensors.
        for (idx, state) in states.iter().enumerate() {
            self.vectorizer()
                .realize_into(state, &mut input_tensors.to_instance_slices(idx));
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
