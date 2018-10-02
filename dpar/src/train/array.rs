use enum_map::EnumMap;

use features::{InputVectorizer, Layer};
use system::{ParserState, TransitionSystem};
use train::InstanceCollector;
use Result;

pub struct ArrayCollector<T> {
    transition_system: T,
    vectorizer: InputVectorizer,
    layers: EnumMap<Layer, Vec<i32>>,
    labels: Vec<usize>,
}

impl<T> ArrayCollector<T> {
    pub fn new(transition_system: T, vectorizer: InputVectorizer) -> Result<Self> {
        Ok(ArrayCollector {
            transition_system: transition_system,
            vectorizer: vectorizer,
            layers: EnumMap::new(),
            labels: Vec::new(),
        })
    }

    pub fn transition_system(&self) -> &T {
        &self.transition_system
    }
}

impl<T> InstanceCollector<T> for ArrayCollector<T>
where
    T: TransitionSystem,
{
    fn collect(&mut self, t: &T::T, state: &ParserState) -> Result<()> {
        let label = self.transition_system.transitions_mut().add(t.clone());
        self.labels.push(label);

        let v = self.vectorizer.realize(state);
        for (layer, layer_vec) in v.layers {
            self.layers[layer].extend(layer_vec);
        }

        Ok(())
    }
}
