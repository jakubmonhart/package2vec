# Package2vec

## Goal
Goal of this project is to create embeddings of python packages in high-dimension for use in classification or regression. For this task we used models created for embedding in NLP, specifically Skip-gram<sup>[1](#skipgram_paper)</sup> and GloVe<sup>[2](#glove_paper)</sup>.

## Data

Using pypi's API, we created dataset of all available dependencies of python packages.

List of all packages installable from pypi: https://pypi.org/simple/ .

Information about individual packages:  https://pypi.org/pypi/'package'/json.

`src/data/parse_requirements.py` parses all available (dependencies of some python packages are not stated within pypi information) dependencies. Sample of parsed dependencies:

``` plaintext
abiflows,custodian
abiflows,fireworks
abiflows,abipy (>=0.7.0)
abilian-core,six  
abilian-core,"Flask (>=1.0,<2.0)"  
abilian-core,Flask-Assets (>=0.12)  
abilian-core,Flask-Babel (>=0.11)  
abilian-core,Flask-Mail (>=0.9.1)  
abilian-core,Flask-Migrate (>=2.0)  
```

First entry in row is python package, second entry is it's dependency. Some dependency entries contain quotation marks, version specification or other unwanted characters, `src/data/filter_requirements.py` cleans the data. It also filters out all packages with only one dependency, as those are not usefull for used models. It's possible to set parameter `MIN_COUNT` to some positive integer to filter out all dependencies which occur less then `MIN_COUNT` in the dataset.

A brief exploration of this dataset can be found in `01-data_exploration.ipynb`.


## Models
Both Skip-gram and GloVe models use co-occurence of words in context window to learn embedding. We treat dependencies of each package as one sentence, single dependency being one word. For this reason, packages with only one dependency are not useful. 

### Skip-gram
The Skip-gram model with negative sampling we used is implemented in `src/models/skipgram.py` using PyTorch library. Using Skip-gram with sentences, context window size must be defined for loss function. Dependencies of one package do not have any particular order and therefore are all treated as one context window irrespective of the number of dependencies.

Configuration of model and data to be used for training are specified in `.yaml` file (sample: `src/model/skipgram_options.yaml`). The model uses `cuda` if available.

Model can be run using `python skipgram.py skipgram_options.yaml`.

Results of this model are described in `02-skipgram_results.ipynb`.

### GloVe
The GloVe model is implemented in `src/models/glove.py` using PyTorch as well. Authors of GloVe paper<sup>[2](#glove_paper)</sup> used 'decreasing weighting function' - word pairs that are farther apart contribute less to the total co-occurence count. In our case, such 'decreasing weighting function' is not necessary, as there are no closer or farther packages in dependencies of one package which we treat as analogy to sentence. We don't define contex window size, instead, all dependencies of one package are treated as context window irrespective of the number of dependencies.

Configuration of model and data to be used for training are specified in `.yaml` file (sample: `src/model/glove_options.yaml`). The model uses `cuda` if available.

Model can be run using `python glove.py glove_options.yaml`.

Results of this model are described in `03-glove_results.ipynb`.

---



<a name="skipgram_paper">1</a>: https://arxiv.org/abs/1310.4546

<a name="glove_paper">2</a>: https://nlp.stanford.edu/pubs/glove.pdf