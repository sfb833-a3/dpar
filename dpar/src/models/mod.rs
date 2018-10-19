pub mod tensorflow;

mod lr;
pub use self::lr::{ConstantLearningRate, ExponentialDecay, LearningRateSchedule};
