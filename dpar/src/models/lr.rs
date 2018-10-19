pub trait LearningRateSchedule {
    fn learning_rate(&self, epoch: usize) -> f32;
}

pub struct ConstantLearningRate(pub f32);

impl LearningRateSchedule for ConstantLearningRate {
    fn learning_rate(&self, _epoch: usize) -> f32 {
        self.0
    }
}

pub struct ExponentialDecay {
    initial_lr: f32,
    decay_rate: f32,
    decay_steps: usize,
    staircase: bool,
}

impl ExponentialDecay {
    pub fn new(initial_lr: f32, decay_rate: f32, decay_steps: usize, staircase: bool) -> Self {
        ExponentialDecay {
            initial_lr,
            decay_rate,
            decay_steps,
            staircase,
        }
    }
}

impl LearningRateSchedule for ExponentialDecay {
    fn learning_rate(&self, epoch: usize) -> f32 {
        let exponent = if self.staircase {
            (epoch / self.decay_steps) as f32
        } else {
            epoch as f32 / self.decay_steps as f32
        };

        self.initial_lr * self.decay_rate.powf(exponent)
    }
}

#[cfg(test)]
mod tests {
    use super::{ConstantLearningRate, ExponentialDecay, LearningRateSchedule};

    #[test]
    pub fn constant_lr() {
        let constant = ConstantLearningRate(0.1);
        assert_relative_eq!(constant.learning_rate(0), 0.1);
        assert_relative_eq!(constant.learning_rate(1), 0.1);
        assert_relative_eq!(constant.learning_rate(5), 0.1);
        assert_relative_eq!(constant.learning_rate(15), 0.1);
        assert_relative_eq!(constant.learning_rate(25), 0.1);
    }

    #[test]
    pub fn exponential_decay_lr() {
        let decay1 = ExponentialDecay::new(0.1, 0.2, 10, true);
        assert_relative_eq!(decay1.learning_rate(0), 0.1);
        assert_relative_eq!(decay1.learning_rate(1), 0.1);
        assert_relative_eq!(decay1.learning_rate(5), 0.1);
        assert_relative_eq!(decay1.learning_rate(15), 0.02);
        assert_relative_eq!(decay1.learning_rate(25), 0.004);

        let decay2 = ExponentialDecay::new(0.1, 0.2, 10, false);
        assert_relative_eq!(decay2.learning_rate(0), 0.1);
        assert_relative_eq!(decay2.learning_rate(1), 0.085133992);
        assert_relative_eq!(decay2.learning_rate(5), 0.044721359);
        assert_relative_eq!(decay2.learning_rate(15), 0.008944271);
        assert_relative_eq!(decay2.learning_rate(25), 0.001788854);
    }
}
