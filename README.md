# Target Word Masking for Location Metonymy Resolution

### Abstract
 Existing metonymy resolution approaches rely on features extracted
  from external resources like dictionaries and hand-crafted lexical
  resources.  In this paper, we propose an end-to-end word-level
  classification approach based only on BERT, without dependencies on
  taggers, parsers, curated dictionaries of place names, or other
  external resources. We show that our approach achieves the
  state-of-the-art on 5 datasets, surpassing conventional BERT models
  and benchmarks by a large margin. We also show that our approach
  generalises well to unseen data.

### Requirements

- python 3.6
- pytorch 1.6
- [transformers 3.4](https://github.com/huggingface/transformers)

### Overview

Our model is based on [`transformers`](https://github.com/huggingface/transformers).

## Metonymy Resolution

We use the code in [this](https://github.com/nlpAThits/WiMCor) repo to get the `Prewin` baselines.

To run our metonymy resolution model, you just need to specify the parameters for `run_metonymy_resolution.py`, see examples in `bash_examples.sh`


## Data

All data mentioned in paper is in `data` directory. For metonymy resolution, datasets are transfered to the same format. 

Note that we use a subset of `WiMCor` and re-split it. 

## Extrinsic Evaluation

#### Edinburgh Geoparser
First download [Edinburgh Geoparser](https://www.ltg.ed.ac.uk/software/geoparser/), use it to detect toponyms, then use `run_metonymy_resolution.py` to classify the readings.

#### NCRF++ 
First download [NCRF++](https://github.com/jiesutd/NCRFpp), modify config file, then train and test.

#### Bert tagger
Run `run_bert_tagger.py` on the data in `data/geoparsing/gold` directly.

#### Our model
Run `run_bert_tagger.py` on the data in `data/geoparsing/gold_entity` to first detect the toponyms, and then convert predictions to json file, finally, use `run_metonymy_resolution.py` to classify the readings.

