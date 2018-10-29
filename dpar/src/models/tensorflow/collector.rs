use enum_map::EnumMap;
use tensorflow::Tensor;

use features::{InputVectorizer, Layer};
use models::tensorflow::{CopyBatches, InstanceSlices, LayerTensors, TensorWrap};
use system::ParserState;
use system::TransitionSystem;
use train::InstanceCollector;

use Result;

pub struct TensorCollector<T> {
    transition_system: T,
    vectorizer: InputVectorizer,
    batch_size: usize,
    inputs: Vec<LayerTensors<i32>>,
    labels: Vec<Tensor<i32>>,
    batch_idx: usize,
}

impl<T> TensorCollector<T> {
    pub fn new(transition_system: T, vectorizer: InputVectorizer, batch_size: usize) -> Self {
        TensorCollector {
            transition_system,
            vectorizer,
            batch_size,
            inputs: Vec::new(),
            labels: Vec::new(),
            batch_idx: 0,
        }
    }

    fn resize_last_batch(&mut self) {
        if self.batch_idx == 0 {
            return;
        }

        let last_size = self.batch_idx;

        let old_inputs = self.inputs.pop().expect("No batches");
        self.inputs.push(old_inputs.copy_batches(last_size as u64));

        let old_labels = self.labels.pop().expect("No batches");
        self.labels.push(old_labels.copy_batches(last_size as u64));
    }

    pub fn into_data(mut self) -> (Vec<Tensor<i32>>, Vec<LayerTensors<i32>>) {
        self.resize_last_batch();

        (self.labels, self.inputs)
    }

    pub fn transition_system(&self) -> &T {
        &self.transition_system
    }

    fn new_layer_tensors(&self, batch_size: usize) -> LayerTensors<i32> {
        let layer_sizes = self.vectorizer.layer_sizes();

        let mut layers: EnumMap<Layer, TensorWrap<i32>> = EnumMap::new();
        for (layer, tensor) in &mut layers {
            *tensor = TensorWrap(Tensor::new(&[batch_size as u64, layer_sizes[layer] as u64]));
        }

        layers
    }
}

impl<T> InstanceCollector<T> for TensorCollector<T>
where
    T: TransitionSystem,
{
    fn collect(&mut self, t: &T::Transition, state: &ParserState) -> Result<()> {
        if self.batch_idx == 0 {
            let layer_tensors = self.new_layer_tensors(self.batch_size);
            self.inputs.push(layer_tensors);
            self.labels.push(Tensor::new(&[self.batch_size as u64]));
        }

        let batch = self.labels.len() - 1;

        let label = self.transition_system.transitions().lookup(t.clone());
        self.labels[batch][self.batch_idx] = label as i32;

        self.vectorizer.realize_into(
            state,
            &mut self.inputs[batch].to_instance_slices(self.batch_idx),
        );

        self.batch_idx += 1;
        if self.batch_idx == self.batch_size {
            self.batch_idx = 0;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tensorflow::Tensor;

    use super::CopyBatches;

    #[test]
    fn copy_batches_test() {
        let original = Tensor::new(&[4, 2])
            .with_values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("Cannot initialize tensor.");
        let copy = original.copy_batches(2);

        assert_eq!(&*copy, &[1.0, 2.0, 3.0, 4.0]);
    }
}
