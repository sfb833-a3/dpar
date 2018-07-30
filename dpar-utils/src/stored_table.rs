use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem;
use std::path::Path;

use dpar::features::{Lookup, LookupTable, MutableLookupTable};
use tensorflow::Tensor;

use {CborRead, CborWrite, Result};

pub enum StoredLookupTable {
    Table(LookupTable),
    FreshTable {
        write: Box<Write>,
        table: MutableLookupTable,
    },
}

impl StoredLookupTable {
    pub fn open<P>(path: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let f = File::open(path)?;
        Ok(StoredLookupTable::Table(LookupTable::from_cbor_read(f)?))
    }

    pub fn open_or_create<P>(path: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let path = &path.as_ref();

        if path.exists() {
            let f = File::open(path)?;
            Ok(StoredLookupTable::Table(LookupTable::from_cbor_read(f)?))
        } else {
            let f = File::create(path)?;
            let write = BufWriter::new(f);
            Ok(StoredLookupTable::FreshTable {
                write: Box::new(write),
                table: MutableLookupTable::new(),
            })
        }
    }
}

impl Drop for StoredLookupTable {
    fn drop(&mut self) {
        if let &mut StoredLookupTable::FreshTable {
            ref mut write,
            ref mut table,
        } = self
        {
            // We want to serialize an immutable table. So, get the inner MutableLookupTable,
            // convert it into a LookupTable.
            let table: LookupTable = mem::replace(table, MutableLookupTable::new()).into();

            if let Err(err) = table.to_cbor_write(write) {
                eprintln!("Error writing lookup table: {}", err);
            }
        }
    }
}

impl Lookup for StoredLookupTable {
    fn embed_matrix(&self) -> Option<&Tensor<f32>> {
        None
    }

    fn lookup(&self, feature: &str) -> Option<usize> {
        match self {
            &StoredLookupTable::Table(ref table) => table.lookup(feature),
            &StoredLookupTable::FreshTable { ref table, .. } => table.lookup(feature),
        }
    }

    fn null(&self) -> usize {
        match self {
            &StoredLookupTable::Table(ref table) => table.null(),
            &StoredLookupTable::FreshTable { ref table, .. } => table.null(),
        }
    }

    fn unknown(&self) -> usize {
        match self {
            &StoredLookupTable::Table(ref table) => table.unknown(),
            &StoredLookupTable::FreshTable { ref table, .. } => table.unknown(),
        }
    }
}
