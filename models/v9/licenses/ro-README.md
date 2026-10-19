---
language: ro
tags:
- bert
- fill-mask
license: mit
---

# bert-base-romanian-cased-v1

The BERT **base**, **cased** model for Romanian, trained on a 15GB corpus, version ![v1.0](https://img.shields.io/badge/v1.0-21%20Apr%202020-ff6666)

### How to use

```python
from transformers import AutoTokenizer, AutoModel
import torch
# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
# tokenize a sentence and run through the model
input_ids = torch.tensor(tokenizer.encode("Acesta este un test.", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
# get encoding
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
```

Remember to always sanitize your text! Replace ``s`` and ``t`` cedilla-letters to comma-letters with :
```
text = text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
```
because the model was **NOT** trained on cedilla ``s`` and ``t``s. If you don't, you will have decreased performance due to ``<UNK>``s and increased number of tokens per word. 

### Evaluation

Evaluation is performed on Universal Dependencies [Romanian RRT](https://universaldependencies.org/treebanks/ro_rrt/index.html) UPOS, XPOS and LAS, and on a NER task based on [RONEC](https://github.com/dumitrescustefan/ronec). Details, as well as more in-depth tests not shown here, are given in the dedicated [evaluation page](https://github.com/dumitrescustefan/Romanian-Transformers/tree/master/evaluation/README.md). 

The baseline is the [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md) model ``bert-base-multilingual-(un)cased``, as at the time of writing it was the only available BERT model that works on Romanian.

| Model                          |  UPOS |  XPOS  |  NER  |  LAS  |
|--------------------------------|:-----:|:------:|:-----:|:-----:|
| bert-base-multilingual-cased   | 97.87 |  96.16 | 84.13 | 88.04 |
| bert-base-romanian-cased-v1    | **98.00** |  **96.46** | **85.88** | **89.69** |

### Corpus 

The model is trained on the following corpora (stats in the table below are after cleaning):

| Corpus    	| Lines(M) 	| Words(M) 	| Chars(B) 	| Size(GB) 	|
|-----------|:--------:|:--------:|:--------:|:--------:|
| OPUS      	|   55.05  	|  635.04  	|   4.045  	|    3.8   	|
| OSCAR     	|   33.56  	|  1725.82 	|  11.411  	|    11    	|
| Wikipedia 	|   1.54   	|   60.47  	|   0.411  	|    0.4   	|
| **Total**     	|   **90.15**  	|  **2421.33** 	|  **15.867**  	|   **15.2**   	|

### Citation

If you use this model in a research paper, I'd kindly ask you to cite the following paper:

```
Stefan Dumitrescu, Andrei-Marius Avram, and Sampo Pyysalo. 2020. The birth of Romanian BERT. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4324–4328, Online. Association for Computational Linguistics.
```

or, in bibtex:

```
@inproceedings{dumitrescu-etal-2020-birth,
    title = "The birth of {R}omanian {BERT}",
    author = "Dumitrescu, Stefan  and
      Avram, Andrei-Marius  and
      Pyysalo, Sampo",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.387",
    doi = "10.18653/v1/2020.findings-emnlp.387",
    pages = "4324--4328",
}
```

#### Acknowledgements

- We'd like to thank [Sampo Pyysalo](https://github.com/spyysalo) from TurkuNLP for helping us out with the compute needed to pretrain the v1.0 BERT models. He's awesome!
