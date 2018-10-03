extern crate conllx;

#[macro_use]
extern crate enum_map;

#[macro_use]
extern crate error_chain;

extern crate failure;

extern crate petgraph;

extern crate serde;

#[macro_use]
extern crate serde_derive;

extern crate tensorflow;

extern crate tf_embed;

#[macro_use]
mod macros;

mod error;
pub use error::*;

pub mod features;

pub mod guide;

pub mod models;

mod numberer;
pub use numberer::Numberer;

pub mod system;

pub mod systems;

pub mod parser;

pub mod train;

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

#[cfg(test)]
#[macro_use]
extern crate maplit;

#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;
