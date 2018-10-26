use enum_map::EnumMap;
use tensorflow::Tensor;

use features::{InputVectorizer, Layer};
use models::tensorflow::{LayerTensors, TensorWrap};
use system::ParserState;
use system::TransitionSystem;
use train::InstanceCollector;

use Result;

pub struct TensorCollector<T> {
    transition_system: T,
    vectorizer: InputVectorizer,
    batch_size: usize,
    inputs: Vec<LayerTensors>,
    labels: Vec<Tensor<i32>>,
    current_inputs: EnumMap<Layer, Vec<i32>>,
    current_labels: Vec<i32>,
}

impl<T> TensorCollector<T> {
    pub fn new(transition_system: T, vectorizer: InputVectorizer, batch_size: usize) -> Self {
        let current_inputs = Self::new_layer_vecs(vectorizer.layer_sizes(), batch_size);

        TensorCollector {
            transition_system,
            vectorizer,
            batch_size,
            inputs: Vec::new(),
            labels: Vec::new(),
            current_inputs,
            current_labels: Vec::with_capacity(batch_size),
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

    /// Create layer tensors with preallocated capacities
    fn new_layer_vecs(
        layer_sizes: EnumMap<Layer, usize>,
        batch_size: usize,
    ) -> EnumMap<Layer, Vec<i32>> {
        let mut layer_vecs: EnumMap<Layer, Vec<i32>> = EnumMap::new();
        for (layer, vec) in &mut layer_vecs {
            vec.reserve(layer_sizes[layer] * batch_size);
        }

        layer_vecs
    }

    pub fn transition_system(&self) -> &T {
        &self.transition_system
    }
}

impl<T> InstanceCollector<T> for TensorCollector<T>
where
    T: TransitionSystem,
{
    fn collect(&mut self, t: &T::Transition, state: &ParserState) -> Result<()> {
        let label = self.transition_system.transitions().lookup(t.clone());

        self.current_labels.push(label as i32);
        self.vectorizer
            .realize_extend(state, &mut self.current_inputs);

        if self.current_labels.len() == self.batch_size {
            self.finalize_batch();
        }

        Ok(())
    }
}
