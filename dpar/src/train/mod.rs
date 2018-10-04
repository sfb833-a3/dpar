use features::InputVectorizer;
use system::{ParserState, TransitionSystem};
use Result;

mod array;
pub use self::array::ArrayCollector;

mod tensor;
pub use self::tensor::TensorCollector;

mod trainer;
pub use self::trainer::GreedyTrainer;

pub trait InstanceCollector<T>
where
    T: TransitionSystem,
{
    fn collect(&mut self, t: &T::T, state: &ParserState) -> Result<()>;
}

pub struct NoopCollector<T> {
    transition_system: T,
    vectorizer: InputVectorizer,
}

impl<T> NoopCollector<T>
where
    T: TransitionSystem,
{
    pub fn new(transition_system: T, vectorizer: InputVectorizer) -> Result<Self> {
        Ok(NoopCollector {
            transition_system: transition_system,
            vectorizer: vectorizer,
        })
    }

    pub fn input_vectorizer(&self) -> &InputVectorizer {
        &self.vectorizer
    }

    pub fn transition_system(&self) -> &T {
        &self.transition_system
    }
}

impl<T> InstanceCollector<T> for NoopCollector<T>
where
    T: TransitionSystem,
{
    fn collect(&mut self, t: &T::T, state: &ParserState) -> Result<()> {
        self.transition_system.transitions_mut().add(t.clone());
        self.vectorizer.realize(state);
        Ok(())
    }
}
