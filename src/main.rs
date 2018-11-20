// Course:      Efficient Linear Algebra and Machine Learning
// Assignment:  Final Assignment, Word Alignment ("Translation Pairs")
// Author:      Michael Watkins
//
// Honor Code:  I pledge that this program represents my own work.

// Michael Watkins, 2018
// Thank you to Theo Varvadoukas for the inspiration:
// https://github.com/tvarvadoukas/ibm_models_1_2/blob/master/build_dictionary-ibm1.py

extern crate clap;

extern crate counter;
use counter::Counter;
use std::io::Write;

extern crate ndarray;
use ndarray::{Array, Axis};

extern crate nlp_tokenize;
use nlp_tokenize::Tokenizer;

extern crate stdinout;
use stdinout::OrExit;

use std::f64::NEG_INFINITY;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::iter::FromIterator;

mod args;
use args::parse_args;

/// Holds a dictionary
struct Dictionary {
    t_table: Array<f64, ndarray::Ix2>, // translation probabilities
    sentence_data: SentenceData,
}

/// Holds sentence data
struct SentenceData {
    pairs: Vec<Vec<Vec<String>>>, // list of sentence pairs

    l1: Vec<String>, // list of words in l1 (source)

    l2: Vec<String>, // list of words in l2 (target)
}

/// Constructs a new Dictionary object
fn build_dictionary(sentence_data: SentenceData, i: usize) -> Dictionary {
    // Translation table to hold l1 -> l2 translation probabilities
    let mut t_table: Array<f64, ndarray::Ix2> = Array::from_elem(
        (sentence_data.l1.len(), sentence_data.l2.len()),
        1. / (sentence_data.l2.len() as f64),
    );

    let mut iterations: usize = 0;
    let mut previous_likelihood: f64 = NEG_INFINITY;
    let mut likelihood: f64 = 0.;

    let eps: f64 = 10.0_f64.powi(-5);
    while (iterations < 2) | ((likelihood - previous_likelihood).abs() > eps) {
        iterations += 1;

        if iterations > 1 {
            previous_likelihood = likelihood
        }

        if iterations > i {
            break;
        }

        // E-step
        likelihood = 0.;
        let mut count_l2_l1: Array<f64, ndarray::Ix2> =
            Array::zeros((sentence_data.l1.len(), sentence_data.l2.len()));
        let mut total_l1: Array<f64, ndarray::Ix1> = Array::zeros(sentence_data.l1.len());

        for pair in &sentence_data.pairs {
            let s_l1 = &pair[0];
            let s_l2 = &pair[1];

            let ind_l1 = s_l1
                .iter()
                .filter_map(|x| sentence_data.l1.iter().position(|y| y == x))
                .collect::<Counter<_>>();

            let ind_l2 = s_l2
                .iter()
                .filter_map(|x| sentence_data.l2.iter().position(|y| y == x))
                .collect::<Counter<_>>();

            let mut l1_sorted = Vec::from_iter(ind_l1.keys());
            l1_sorted.sort_unstable();

            let mut l2_sorted = Vec::from_iter(ind_l2.keys());
            l2_sorted.sort_unstable();

            let len = (l1_sorted.len(), l2_sorted.len());
            let l1_sorted = Array::from_vec(l1_sorted).into_shape((len.0, 1)).unwrap();
            let l2_sorted = Array::from_vec(l2_sorted).into_shape((1, len.1)).unwrap();

            let indexes = Array::from_vec(vec![l1_sorted.to_owned(), l2_sorted.to_owned()]);

            // Submatrix of t_table focusing on current alignment possibilities
            let mut t_subtable = Array::<f64, ndarray::Ix2>::zeros((len.0, len.1));
            let mut t_sub_to_full_idx =
                Array::<(usize, usize), ndarray::Ix2>::from_elem((len.0, len.1), (0, 0));

            // TODO: Check on this construction and see if it's actually working as intended.
            for i in 0..indexes[0].len() {
                for j in 0..indexes[1].len() {
                    t_subtable[[i, j]] =
                        t_table[[indexes[0][[i, 0]].to_owned(), indexes[1][[0, j]].to_owned()]];
                    t_sub_to_full_idx[[i, j]] =
                        (indexes[0][[i, 0]].to_owned(), indexes[1][[0, j]].to_owned());
                }
            }

            for i in 0..l2_sorted.len() {
                let y = l2_sorted[[0, i]];
                if ind_l2[y] > 1 {
                    for row in 0..t_subtable.rows() {
                        t_subtable[[row, i]] *= ind_l2[y] as f64;
                    }
                }
            }

            for i in 0..l1_sorted.len() {
                let y = l1_sorted[[i, 0]];
                if ind_l1[y] > 1 {
                    for col in 0..t_subtable.cols() {
                        t_subtable[[i, col]] *= ind_l1[y] as f64;
                    }
                }
            }

            // Compute normalization constant for each word l1.
            let z = t_subtable.sum_axis(Axis(0));

            // Update log-likelihood
            for i in z.iter() {
                likelihood += i.ln();
            }

            // collect counts
            let mut temp = t_subtable.to_owned();

            for i in 0..temp.shape()[0] {
                for j in 0..temp.shape()[1] {
                    temp[[i, j]] /= z[j];
                }
            }

            for i in 0..indexes[0].len() {
                for j in 0..indexes[1].len() {
                    // println!("{:?},{:?}", i, j);
                    count_l2_l1[[t_sub_to_full_idx[[i, j]].0, t_sub_to_full_idx[[i, j]].1]] +=
                        temp[[i, j]];
                }
            }

            let temp_sums = temp.sum_axis(Axis(1));
            for i in 0..l1_sorted.len() {
                total_l1[*l1_sorted[[i, 0]]] += temp_sums[i];
            }
        }

        t_table = (count_l2_l1.t().to_owned() / total_l1).t().to_owned();
        println!(
            "{:?}\tNew: {:4}\tOld:{:4}\n",
            iterations, likelihood, previous_likelihood
        );
    }

    Dictionary {
        t_table: t_table,
        sentence_data: sentence_data,
    }
}

// Load sentences
fn process_sentences(source_file: &str, target_file: &str) -> SentenceData {
    let source_file = File::open(source_file).or_exit("Cannot open source file", 1);
    let target_file = File::open(target_file).or_exit("Cannot open target file", 1);

    let mut pairs: Vec<Vec<Vec<String>>> = Vec::new();

    // let mut l1: HashSet<String> = HashSet::new();
    let mut l1: Vec<String> = Vec::new();

    // let mut l2: HashSet<String> = HashSet::new();
    let mut l2: Vec<String> = Vec::new();

    for (idx, line) in BufReader::new(&source_file).lines().enumerate() {
        // might want to check this .unwrap() later on...
        let mut sentence: Vec<String> = tokenize(line.unwrap().to_lowercase());
        pairs.push(Vec::new());
        sentence.push(String::from("null"));
        pairs[idx].push(sentence.to_owned());

        for tok in sentence {
            if !l1.contains(&tok) {
                l1.push(tok);
            }
        }
    }

    for (idx, line) in BufReader::new(&target_file).lines().enumerate() {
        // might want to check this .unwrap() later on...
        let line = line.unwrap().to_lowercase();
        let sentence: Vec<String> = tokenize(line);
        pairs[idx].push(sentence.to_owned());

        for tok in sentence {
            if !l2.contains(&tok) {
                l2.push(tok);
            }
        }
    }

    SentenceData {
        pairs: pairs,
        l1: l1,
        l2: l2,
    }
}

// Tokenize sentences
fn tokenize(sentence: String) -> Vec<String> {
    let tokenizer = nlp_tokenize::WhitespaceTokenizer::new();

    let mut toks: Vec<String> = Vec::new();
    let tok_bounds = tokenizer.tokenize(&sentence);

    for bound in tok_bounds {
        toks.push(String::from(&sentence[bound.0..bound.1]));
    }

    toks
}

// Save the dictionary to a file
fn save_dictionary(dictionary: Dictionary, output_file: &str, p: f64) {
    let output = File::create(output_file).expect("Cannot open output file");
    let mut bufwriter = BufWriter::new(output);

    let mut out: String = String::from("");
    let mut col = 0;
    let mut row = 0;
    let len = dictionary.sentence_data.l2.len();
    let l1 = dictionary.sentence_data.l1.to_owned();
    let l2 = dictionary.sentence_data.l2.to_owned();

    for cell in dictionary.t_table.iter() {
        if col >= len {
            col = 0;
            row += 1;
        }

        if cell > &p {
            println!("{:?}\t{:?}\t{:?}", l1[row], l2[col], cell);
            out = format!(
                "{}{}",
                out,
                format!("{:?}\t{:?}\t{:4}\n", l1[row], l2[col], cell)
            );
        }

        col += 1;
    }

    bufwriter
        .write_all(out.as_bytes())
        .expect("Could not write to output file");
}

fn main() {
    let arguments = parse_args();

    let source_file = arguments.value_of("SOURCE").unwrap();
    let target_file = arguments.value_of("TARGET").unwrap();
    let output_file = arguments.value_of("OUTPUT").unwrap();

    let p: f64 = arguments
        .value_of("PROBABILITY")
        .unwrap_or("0.5")
        .parse()
        .unwrap();

    let i: usize = arguments
        .value_of("ITERATIONS")
        .unwrap_or("30")
        .parse()
        .unwrap();

    let sentence_data = process_sentences(source_file, target_file);

    let dictionary = build_dictionary(sentence_data, i);

    save_dictionary(dictionary, output_file, p);
}
