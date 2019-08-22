use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use failure::Error;
use rust2vec::{
    embeddings::Embeddings as R2VEmbeddings, io::ReadEmbeddings, storage::StorageWrap,
    vocab::VocabWrap,
};

use dpar::features::Embeddings;

/// Read association strengths for dependency triples from a text file.
///
/// Such a text file consists of lines with the tab-separated format
///
/// ~~~text,no_run
/// [token+] [token+] [deprel+] association_strength
/// ~~~
pub fn associations_from_buf_read(
    f: File,
) -> Result<HashMap<(String, String, String), f32>, Error> {
    let mut association_strengths: HashMap<(String, String, String), f32> = HashMap::new();
    for l in BufReader::new(f).lines() {
        let l = l.unwrap();
        let line = l.split("\t").collect::<Vec<_>>();
        association_strengths.insert(
            (
                line[0].to_string(),
                line[1].to_string(),
                line[2].to_string(),
            ),
            line[3].parse::<f32>().unwrap(),
        );
    }
    Ok(association_strengths)
}

/// Read in focus and context embeddings on the basis of which the
/// association strength between two `AttachmentAddr`s will be calculated.
pub fn embeds_from_files(
    focus_embeds: File,
    context_embeds: File,
) -> Result<(Embeddings, Embeddings), Error> {
    let mut focus_reader = BufReader::new(&focus_embeds);
    let focus_embeds: R2VEmbeddings<VocabWrap, StorageWrap> =
        R2VEmbeddings::read_embeddings(&mut focus_reader).unwrap();

    let mut context_reader = BufReader::new(&context_embeds);
    let context_embeds: R2VEmbeddings<VocabWrap, StorageWrap> =
        R2VEmbeddings::read_embeddings(&mut context_reader).unwrap();

    Ok((
        Embeddings::from(focus_embeds),
        Embeddings::from(context_embeds),
    ))
}

