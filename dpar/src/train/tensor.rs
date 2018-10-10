use enum_map::EnumMap;
use tensorflow::Tensor;

use features::{InputVectorizer, Layer};
use models::tensorflow::{LayerTensors, TensorWrap};
use system::ParserState;
use system::TransitionSystem;
use train::InstanceCollector;

use Result;

/// TODO: handle last batch, typically incomplete.
pub struct TensorCollector<T> {
    transition_system: T,
    vectorizer: InputVectorizer,
    batch_size: usize,
    inputs: Vec<LayerTensors>,
    labels: Vec<Tensor<i32>>,
    current_inputs: EnumMap<Layer, Vec<i32>>,
    current_labels: Vec<i32>,
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
            inputs: Vec::new(),
            labels: Vec::new(),
            current_inputs: EnumMap::new(),
            current_labels: Vec::new(),
            is_training,
        }
    }

    fn finalize_batch(&mut self) {
        let batch_size = self.current_labels.len();
        if batch_size == 0 {
            return;
        }

        let label_tensor = Tensor::new(&[batch_size as u64])
            .with_values(&self.current_labels)
            .expect("Incorrect label batch size.");

        let mut input_tensors = EnumMap::new();
        for (layer, vec) in &self.current_inputs {
            let input_len = vec.len() / batch_size;
            let tensor = Tensor::new(&[batch_size as u64, input_len as u64])
                .with_values(&vec)
                .expect("Incorrect inputs shape.");
            input_tensors[layer] = TensorWrap(tensor);
        }

        self.labels.push(label_tensor);
        self.inputs.push(LayerTensors(input_tensors));

        self.current_labels.clear();
        for vec in self.current_inputs.values_mut() {
            vec.clear();
        }
    }

    pub fn into_data(mut self) -> (Vec<Tensor<i32>>, Vec<LayerTensors>) {
        self.finalize_batch();

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
        let label = if self.is_training {
            self.transition_system.transitions_mut().add(t.clone())
        } else {
            self.transition_system.transitions().number(t).unwrap_or(0)
        };

        self.current_labels.push(label as i32);
        self.vectorizer
            .realize_extend(state, &mut self.current_inputs);

        if self.current_labels.len() == self.batch_size {
            self.finalize_batch();
        }

        Ok(())
    }
}
