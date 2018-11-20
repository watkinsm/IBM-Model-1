// Course:      Efficient Linear Algebra and Machine Learning
// Assignment:  Final Assignment, Word Alignment ("Translation Pairs")
// Author:      Michael Watkins
//
// Honor Code:  I pledge that this program represents my own work.

use clap::{App, AppSettings, Arg, ArgMatches};

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

pub fn parse_args() -> ArgMatches<'static> {
    App::new("word-alignment")
        .settings(DEFAULT_CLAP_SETTINGS)
        .arg(
            Arg::with_name("ITERATIONS")
                .short("i")
                .long("iterations")
                .value_name("I")
                .help("Maximum iterations for EM algorithm (default: 30)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("PROBABILITY")
                .short("p")
                .long("probability")
                .value_name("I")
                .help("Minimum probability for a word alignment (default: 0.5)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("SOURCE")
                .help("Source language sentence alignments")
                .index(1)
                .required(true),
        )
        .arg(
            Arg::with_name("TARGET")
                .help("Source language sentence alignments")
                .index(2)
                .required(true),
        )
        .arg(
            Arg::with_name("OUTPUT")
                .help("Output file to store word alignments")
                .index(3)
                .required(true),
        )
        .get_matches()
}
