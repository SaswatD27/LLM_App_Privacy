# LLM Application Privacy
## Overview
Readme to be updated.

For a sample run, use the following command. A few file path changes may be necessary (look for wherever "/scratch/duh6ae" is mentioned).

```
python /src/main.py
```

Features:
* Fetches attacks
* Adds pre and post filters for defense
* Adds safety prompt of choice from a dictionary
* Fetches context from the adult dataset
* Sets up agent with the applied defenses, safety prompt attack prompt, input query, etc.

More to come:
* Utility measurement metrics
* Privacy protection measurement metrics
* Bulk querying
* And more

## Files
* /local_data: Contains datasets that are available offline.
* /src/main.py: Main executable for an example run with a safety_prompt, a Cognitive Hacking attack prompt, a query about the person's race (from the adult dataset), a query prefilter (rewriter), and postfilter (checks for forbidden words in the response).
* /src/utils: Contains files for instantiating the agent and loaders for attacks, defenses, and contexts.
* /src/utils/agentUtils.py: Contains the class agent; this includes the whole application and any applied defenses and attacks
* /src/utils/attack: Contains attack dictionary and attack loader
* /src/utils/defense: Contains defense dictionary, defense definition, and defense loader
* /src/utils/contextLoading: Contains utils to load a context into agent from a Huggingface dataset
