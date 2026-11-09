# Internal Knowledge-Base Q&A

Ask a policy question and get an answer that cites the file and chunk it came from, with retrieval done locally by MiniLM embeddings and a numpy cosine index, no hosted vector service. Portfolio, Applied AI, by Maximus Friedrich Baluyot.

**Watch first:** [the 90-second film](https://github.com/Keshigami/internal_kb_qa/blob/main/demo/film.mp4).

## Pages
- [Getting started](Getting-Started.md): run it in three commands, what you see, the API.
- [How it was built](How-It-Was-Built.md): from the brief to a running service, step by step.
- [Results](Results.md): the numbers and what they mean.
- [Limits and next steps](Limits-and-Next-Steps.md): what it cannot do yet.
- [The film](The-Film.md): what the four chapters show.

## The brief
Employees lose time hunting through policy PDFs and wikis for a number or a deadline, and the answers they get from
colleagues are often wrong. This app indexes the policy documents, finds the passages that match a question and
answers with citations so the reader can verify every fact. Solved means a question about expenses, leave, security
or remote work returns the right section in under a second, with the source next to it.

## The one-line answer
4/4 · sample questions · right section on top · MiniLM

## One honest limit
Offline, the answer is the passage, not prose.
