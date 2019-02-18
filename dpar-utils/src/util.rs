use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;

use failure::Error;

/// Read association strengths for dependency triples from text files in
/// a directory or from a single file.
///
/// A text file consists of lines with the tab-separated format
///
/// ~~~text,no_run
/// [token+] [token+] association_strength
/// ~~~
///
/// The file is named with the dependency relation of which head-dependent
/// pairs are listed with their association measure in the file.
///
/// E.g. the file is called `SUBJ.assoc` and contains all tokens related via the
/// SUBJ relation and their association measure.
pub fn associations_from_files(
    dir: &Path,
) -> Result<HashMap<(String, String, String), f32>, Error> {
    let mut association_strengths: HashMap<(String, String, String), f32> = HashMap::new();

    if dir.is_dir() {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_file() {
                let f = File::open(&path)?;
                for l in BufReader::new(f).lines() {
                    let l = l.unwrap();
                    let line = l.split("\t").collect::<Vec<_>>();
                    association_strengths.insert(
                        (
                            line[0].to_string(),
                            line[1].to_string(),
                            path.file_stem().unwrap().to_string_lossy().to_string(),
                        ),
                        line[2].parse::<f32>().unwrap(),
                    );
                }
            }
        }
    } else if dir.is_file() {
        let f = File::open(&dir)?;
        for l in BufReader::new(f).lines() {
            let l = l.unwrap();
            let line = l.split("\t").collect::<Vec<_>>();
            association_strengths.insert(
                (
                    line[0].to_string(),
                    line[1].to_string(),
                    dir.file_stem().unwrap().to_string_lossy().to_string(),
                ),
                line[2].parse::<f32>().unwrap(),
            );
        }
    }

    Ok(association_strengths)
}
