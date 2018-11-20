# IBM-Model-1

This model requires two files representing sentence alignments from one language to another.

ex)
* file 1, line 1: Das ist ein Handy .
* file 2, line 1: This is a cellphone .

Run the program using:
$ cargo run \[l1_filename.txt] \[l2_filename.txt] \[output_filename.tsv]

ex) $ cargo run de_en.de.txt de_en.en.txt de_en.dictionary.tsv

The output will be a tab-separated file consisting of the source word, the target word, and the probability that the target word is a translation of the source word:
* "unserer" "our" 0.9996887505447988
* "sogar" "even"  0.99876209969135
