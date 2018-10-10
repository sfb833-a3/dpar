extern crate conllx;
extern crate dpar;
#[macro_use]
extern crate dpar_utils;
extern crate getopts;
extern crate indicatif;
extern crate stdinout;
extern crate tensorflow;

use std::env::args;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process;

use conllx::{HeadProjectivizer, Projectivize, ReadSentence};
use dpar::features::InputVectorizer;
use dpar::models::tensorflow::{LayerTensors, TensorflowModel};
use dpar::system::{sentence_to_dependencies, ParserState};
use dpar::systems::{
    ArcEagerSystem, ArcHybridSystem, ArcStandardSystem, StackProjectiveSystem, StackSwapSystem,
};
use dpar::train::{GreedyTrainer, TensorCollector};
use getopts::Options;
use indicatif::{ProgressBar, ProgressStyle};
use tensorflow::Tensor;

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
    let (train_labels, train_inputs) = collect_data(&config, reader, true).or_exit();

    let input_file = File::open(&matches.free[2]).or_exit();
    let reader = conllx::Reader::new(BufReader::new(FileProgress::new(input_file)));
    eprintln!("Vectorizing validation data...");
    let (validation_labels, validation_inputs) = collect_data(&config, reader, false).or_exit();

    train(
        &config,
        train_labels,
        train_inputs,
        validation_labels,
        validation_inputs,
    ).or_exit();
}

fn train(
    config: &Config,
    train_labels: Vec<Tensor<i32>>,
    train_inputs: Vec<LayerTensors>,
    validation_labels: Vec<Tensor<i32>>,
    validation_inputs: Vec<LayerTensors>,
) -> Result<()> {
    match config.parser.system.as_ref() {
        "arceager" => train_with_system::<ArcEagerSystem>(
            config,
            train_labels,
            train_inputs,
            validation_labels,
            validation_inputs,
        ),
        "archybrid" => train_with_system::<ArcHybridSystem>(
            config,
            train_labels,
            train_inputs,
            validation_labels,
            validation_inputs,
        ),
        "arcstandard" => train_with_system::<ArcStandardSystem>(
            config,
            train_labels,
            train_inputs,
            validation_labels,
            validation_inputs,
        ),
        "stackproj" => train_with_system::<StackProjectiveSystem>(
            config,
            train_labels,
            train_inputs,
            validation_labels,
            validation_inputs,
        ),
        "stackswap" => train_with_system::<StackSwapSystem>(
            config,
            train_labels,
            train_inputs,
            validation_labels,
            validation_inputs,
        ),
        _ => {
            stderr!("Unsupported transition system: {}", config.parser.system);
            process::exit(1);
        }
    }
}

fn train_with_system<S>(
    config: &Config,
    train_labels: Vec<Tensor<i32>>,
    train_inputs: Vec<LayerTensors>,
    validation_labels: Vec<Tensor<i32>>,
    validation_inputs: Vec<LayerTensors>,
) -> Result<()>
where
    S: SerializableTransitionSystem,
{
    let system = S::default();
    let lookups = config.lookups.load_lookups()?;
    let inputs = config.parser.load_inputs()?;
    let vectorizer = InputVectorizer::new(lookups, inputs);
    let mut model = TensorflowModel::load_graph(
        &config.model.config_to_protobuf().or_exit(),
        &config.model.model_to_protobuf().or_exit(),
        system,
        vectorizer,
        &config.lookups.layer_ops(),
    )?;

    for i in 0..10 {
        run_epoch(&mut model, &train_labels, &train_inputs, i, true);
        run_epoch(&mut model, &validation_labels, &validation_inputs, i, false);
        model.save(format!("epoch-{}", i)).or_exit();
    }

    Ok(())
}

fn run_epoch<S>(
    model: &mut TensorflowModel<S>,
    labels: &[Tensor<i32>],
    inputs: &[LayerTensors],
    epoch: usize,
    is_training: bool,
)
where
    S: SerializableTransitionSystem,
{
        let epoch_type = if is_training {
            "train"
        } else {
            "validation"
        };

        let mut instances = 0;
        let mut loss = 0f32;
        let mut acc = 0f32;

        let progress = ProgressBar::new(labels.len() as u64);
        progress.set_style(ProgressStyle::default_bar().template(&format!("{{bar}} {} batch {{pos}}/{{len}}", epoch_type)));
        for (labels, inputs) in labels.iter().zip(inputs.iter()) {
            let (batch_loss, batch_acc) = if is_training {
                model.train(inputs, labels)
            } else {
                model.validate(inputs, labels)
            };

            loss += batch_loss * labels.dims()[0] as f32;
            acc += batch_acc * labels.dims()[0] as f32;
            instances += labels.dims()[0];
            progress.inc(1);
        }
        progress.finish();

        loss /= instances as f32;
        acc /= instances as f32;

        eprintln!("Epoch {} ({}): loss: {}, acc: {}", epoch_type, epoch, loss, acc);
}

fn collect_data<R>(
    config: &Config,
    reader: conllx::Reader<R>,
    is_training: bool,
) -> Result<(Vec<Tensor<i32>>, Vec<LayerTensors>)>
where
    R: BufRead,
{
    match config.parser.system.as_ref() {
        "arceager" => collect_with_system::<R, ArcEagerSystem>(config, reader, is_training),
        "archybrid" => collect_with_system::<R, ArcHybridSystem>(config, reader, is_training),
        "arcstandard" => collect_with_system::<R, ArcStandardSystem>(config, reader, is_training),
        "stackproj" => collect_with_system::<R, StackProjectiveSystem>(config, reader, is_training),
        "stackswap" => collect_with_system::<R, StackSwapSystem>(config, reader, is_training),
        _ => {
            stderr!("Unsupported transition system: {}", config.parser.system);
            process::exit(1);
        }
    }
}

fn collect_with_system<R, S>(
    config: &Config,
    reader: conllx::Reader<R>,
    is_training: bool,
) -> Result<(Vec<Tensor<i32>>, Vec<LayerTensors>)>
where
    R: BufRead,
    S: SerializableTransitionSystem,
{
    let lookups = config.lookups.load_lookups()?;
    let inputs = config.parser.load_inputs()?;
    let vectorizer = InputVectorizer::new(lookups, inputs);
    let system: S = load_transition_system_or_new(&config)?;
    let collector = TensorCollector::new(
        system,
        vectorizer,
        config.parser.train_batch_size,
        is_training,
    );
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

    write_transition_system(&config, trainer.collector().transition_system())?;

    Ok(trainer.into_collector().into_data())
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
