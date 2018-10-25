use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::Debug;
use std::hash::Hash;

use serde::de::DeserializeOwned;
use serde::{Serialize, Serializer};

use guide::Guide;
use numberer::Numberer;
use system::{DependencySet, ParserState};

pub trait TransitionSystem {
    type Transition: Transition;
    type Oracle: Guide<Transition = Self::Transition>;

    fn is_terminal(state: &ParserState) -> bool;
    fn oracle(gold_dependencies: &DependencySet) -> Self::Oracle;
    fn transitions(&self) -> &Transitions<Self::Transition>;
}

pub trait Transition: Clone + Debug + Eq + Hash + Serialize + DeserializeOwned {
    type S: TransitionSystem;

    fn is_possible(&self, state: &ParserState) -> bool;
    fn apply(&self, state: &mut ParserState);
}

#[derive(Deserialize, Eq, PartialEq)]
pub enum Transitions<T>
where
    T: Eq + Hash,
{
    Fresh(RefCell<Numberer<T>>),
    Frozen(Numberer<T>),
}

impl<T> Serialize for Transitions<T>
where
    T: Eq + Hash + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use self::Transitions::*;

        match self {
            Fresh(refcell) => {
                serializer.serialize_newtype_variant("Transitions", 1, "Frozen", &*refcell.borrow())
            }
            Frozen(ref numberer) => {
                serializer.serialize_newtype_variant("Transitions", 1, "Frozen", numberer)
            }
        }
    }
}

impl<T> Transitions<T>
where
    T: Clone + Eq + Hash,
{
    pub fn len(&self) -> usize {
        use self::Transitions::*;

        match self {
            Fresh(cell) => cell.borrow().len() + 1,
            Frozen(numberer) => numberer.len() + 1,
        }
    }

    pub fn lookup(&self, t: T) -> usize {
        use self::Transitions::*;

        match self {
            Fresh(cell) => cell.borrow_mut().add(t),
            Frozen(numberer) => numberer.number(&t).unwrap_or(0),
        }
    }

    pub fn value(&self, number: usize) -> Option<Cow<T>> {
        use self::Transitions::*;

        match self {
            Fresh(cell) => cell.borrow().value(number).cloned().map(Cow::Owned),
            Frozen(numberer) => numberer.value(number).map(Cow::Borrowed),
        }
    }
}

impl<T> Default for Transitions<T>
where
    T: Clone + Eq + Hash,
{
    fn default() -> Self {
        Transitions::Fresh(RefCell::new(Numberer::new(1)))
    }
}
