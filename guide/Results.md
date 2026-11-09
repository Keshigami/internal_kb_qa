# Results

On this machine: 18 chunks from 4 files, MiniLM, index build about 15 s including model load, search under 50 ms.
Retrieval on the four sample questions is 4/4 (top chunk is the right section for leave of absence, expense
submission, stolen laptop, working abroad), checked by the tests with TF-IDF and by hand with MiniLM. This is a
sanity check, not the >85% precision claim: that needs a labelled question set of a few hundred. The hallucination
rate cannot be measured without a key; the prompt restricts the model to the excerpts and the page flags answers
without citation markers.
