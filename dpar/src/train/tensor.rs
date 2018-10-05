use enum_map::EnumMap;
use tensorflow::Tensor;

use features::InputVectorizer;
use models::tensorflow::LayerTensors;
use system::ParserState;
use system::TransitionSystem;
use train::InstanceCollector;

use Result;

/// TODO: handle last batch, typically incomplete.
pub struct TensorCollector<T> {
    transition_system: T,
    vectorizer: InputVectorizer,
    batch_size: usize,
    batch_idx: usize,
    inputs: Vec<LayerTensors>,
    labels: Vec<Tensor<i32>>,
    is_training: bool,
}

impl<T> TensorCollector<T> {
    pub fn new(
        transition_system: T,
        vectorizer: InputVectorizer,
        batch_size: usize,
        is_training: bool,
    ) -> Self {
        TensorCollector {
            transition_system,
            vectorizer,
            batch_size,
            batch_idx: 0,
            inputs: Vec::new(),
            labels: Vec::new(),
            is_training,
        }
    }

    // Return the current batch, creating it if it doesn't exist.
    fn ensure_batch(&mut self) -> usize {
        if self.batch_idx == 0 {
            self.labels.push(Tensor::new(&[self.batch_size as u64]));

            let mut input_batch = LayerTensors(EnumMap::new());

            for (layer, size) in self.vectorizer.layer_sizes() {
                input_batch[layer] = Tensor::new(&[self.batch_size as u64, size as u64]).into();
            }

            self.inputs.push(input_batch);
        }

        self.labels.len() - 1
    }

    pub fn into_data(self) -> (Vec<Tensor<i32>>, Vec<LayerTensors>) {
        (self.labels, self.inputs)
    }

    pub fn transition_system(&self) -> &T {
        &self.transition_system
    }
}

impl<T> InstanceCollector<T> for TensorCollector<T>
where
    T: TransitionSystem,
{
    fn collect(&mut self, t: &T::T, state: &ParserState) -> Result<()> {
        let batch = self.ensure_batch();

        let label = if self.is_training {
            self.transition_system.transitions_mut().add(t.clone())
        } else {
            self.transition_system.transitions().number(t).unwrap_or(0)
        };

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
