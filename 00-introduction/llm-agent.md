
# AI Agent

## 


### History of LLM Agents

<img src="image.png" alt="alt text" width="650">


#### RAG for Knowledge

Retrieval-Augmented Generation: answer knowledge-intensive questions with extra corpora/a retriever.

We have a retriever-based NLP.

<img src="image-1.png" alt="alt text" width="800">

- what if there's no corpora? (e.g. who is the latest PM?)

#### Tool use

Introduce special tokens to invoke tool calls for search engines, calculators, etc.
- Unnatural format requires task/tool-specific fine-tuning~
- Multiple tool call

<img src="image-2.png" alt="alt text" width="450">

#### What if both needed?

We have the two paradigm, reasoning and acting. Now a new paragigm ReAct generates both for reasoning and actiing.

<img src="image-3.png" alt="alt text" width="550">

Synergy: acting supports reasoning, reasoning guides acting. 

<img src="image-4.png" alt="alt text" width="750">

Acting is helping reasoning to get real-time information. Also, reasoning is constantly guiding the acting to plan the situation and re-lan the situation based on exceptions. For example:

<img src="image-5.png" alt="alt text" width="750">













