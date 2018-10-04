extern crate conllx;
extern crate dpar;
#[macro_use]
extern crate dpar_utils;
extern crate getopts;
extern crate stdinout;

use std::env::args;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process;

use conllx::{HeadProjectivizer, Projectivize, ReadSentence};
use dpar::features::InputVectorizer;
use dpar::system::{sentence_to_dependencies, ParserState};
use dpar::systems::{
    ArcEagerSystem, ArcHybridSystem, ArcStandardSystem, StackProjectiveSystem, StackSwapSystem,
};
use dpar::train::{GreedyTrainer, TensorCollector};
use getopts::Options;

use dpar_utils::{Config, FileProgress, OrExit, Result, SerializableTransitionSystem, TomlRead};

fn print_usage(program: &str, opts: Options) {
    let brief = format!(
        "Usage: {} [options] CONFIG TRAIN_DATA VALID_DATA OUTPUT.HDF5",
        program
    );
    print!("{}", opts.usage(&brief));
}

fn main() {
    let args: Vec<String> = args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optflag("h", "help", "print this help menu");
    let matches = opts.parse(&args[1..]).or_exit();

    if matches.opt_present("h") {
        print_usage(&program, opts);
        return;
    }

    if matches.free.len() != 4 {
        print_usage(&program, opts);
        return;
    }

    let config_file = File::open(&matches.free[0]).or_exit();
    let mut config = Config::from_toml_read(config_file).or_exit();
    config.relativize_paths(&matches.free[0]).or_exit();

    let input_file = File::open(&matches.free[1]).or_exit();
    let reader = conllx::Reader::new(BufReader::new(FileProgress::new(input_file)));
    eprintln!("Vectorizing training data...");
    train(&config, reader).or_exit();

    let input_file = File::open(&matches.free[2]).or_exit();
    let reader = conllx::Reader::new(BufReader::new(FileProgress::new(input_file)));
    eprintln!("Vectorizing validation data...");
    train(&config, reader).or_exit();
}

fn train<R>(config: &Config, reader: conllx::Reader<R>) -> Result<()>
where
    R: BufRead,
{
    match config.parser.system.as_ref() {
        "arceager" => train_with_system::<R, ArcEagerSystem>(config, reader),
        "archybrid" => train_with_system::<R, ArcHybridSystem>(config, reader),
        "arcstandard" => train_with_system::<R, ArcStandardSystem>(config, reader),
        "stackproj" => train_with_system::<R, StackProjectiveSystem>(config, reader),
        "stackswap" => train_with_system::<R, StackSwapSystem>(config, reader),
        _ => {
            stderr!("Unsupported transition system: {}", config.parser.system);
            process::exit(1);
        }
    }
}

fn train_with_system<R, S>(config: &Config, reader: conllx::Reader<R>) -> Result<()>
where
    R: BufRead,
    S: SerializableTransitionSystem,
{
    let lookups = config.lookups.load_lookups()?;
    let inputs = config.parser.load_inputs()?;
    let vectorizer = InputVectorizer::new(lookups, inputs);
    let system: S = load_transition_system_or_new(&config)?;
    let collector = TensorCollector::new(system, vectorizer, config.parser.train_batch_size);
    let mut trainer = GreedyTrainer::new(collector);
    let projectivizer = HeadProjectivizer::new();

    for sentence in reader.sentences() {
        let sentence = if config.parser.pproj {
            projectivizer.projectivize(&sentence?)?
        } else {
            sentence?
        };

        let dependencies = sentence_to_dependencies(&sentence).or_exit();

        let mut state = ParserState::new(&sentence);
        trainer.parse_state(&dependencies, &mut state)?;
    }

    write_transition_system(&config, trainer.collector().transition_system())
}

fn load_transition_system_or_new<T>(config: &Config) -> Result<T>
where
    T: SerializableTransitionSystem,
{
    let transitions_path = Path::new(&config.parser.transitions);
    if !transitions_path.exists() {
        return Ok(T::default());
    }

    println!("Loading transitions from: {:?}", transitions_path);

    let f = File::open(transitions_path)?;
    let system = T::from_cbor_read(f)?;

    Ok(system)
}

fn write_transition_system<T>(config: &Config, system: &T) -> Result<()>
where
    T: SerializableTransitionSystem,
{
    let transitions_path = Path::new(&config.parser.transitions);
    if transitions_path.exists() {
        return Ok(());
    }

    let mut f = File::create(transitions_path)?;
    system.to_cbor_write(&mut f)?;
    Ok(())
}
