# Limits and next steps

- Offline, the answer is the best passage, not generated text; a key turns on the written, cited answer.
- Four fictional documents; a real deployment points `KB_DOCS_DIR` at the policy folder and rebuilds.
- Scores are not calibrated: an off-topic question still returns passages, and only the prompt stops the model from
  answering from them.
- Text only; tables and images in PDFs are skipped. A reranker and a labelled eval set are the next steps.
