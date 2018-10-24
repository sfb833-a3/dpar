use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Deref, DerefMut};

use serde::de::DeserializeOwned;
use serde::Serialize;

use guide::Guide;
use numberer::Numberer;
use system::{DependencySet, ParserState};

pub trait TransitionSystem {
    type Transition: Transition;
    type Oracle: Guide<Transition = Self::Transition>;

    fn is_terminal(state: &ParserState) -> bool;
    fn oracle(gold_dependencies: &DependencySet) -> Self::Oracle;
    fn transitions(&self) -> &Transitions<Self::Transition>;
    fn transitions_mut(&mut self) -> &mut Transitions<Self::Transition>;
}

pub trait Transition: Clone + Debug + Eq + Hash + Serialize + DeserializeOwned {
    type S: TransitionSystem;

    fn is_possible(&self, state: &ParserState) -> bool;
    fn apply(&self, state: &mut ParserState);
}

#[derive(Eq, PartialEq, Serialize, Deserialize)]
pub struct Transitions<T>(Numberer<T>)
where
    T: Eq + Hash;

impl<T> Transitions<T>
where
    T: Clone + Eq + Hash,
{
    pub fn len(&self) -> usize {
        self.0.len() + 1
    }

    pub fn null(&self) -> usize {
        0
    }
}

impl<T> Default for Transitions<T>
where
    T: Transition,
{
    fn default() -> Self {
        Transitions(Numberer::new(1))
    }
}

impl<T> Deref for Transitions<T>
where
    T: Transition,
{
    type Target = Numberer<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Transitions<T>
where
    T: Transition,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
