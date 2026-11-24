# Interview preparation

About 100 questions for LLM application engineer interviews, in 12 areas. Every answer has the same parts:

- **Short answer:** what to say in the first 30 seconds.
- **Deeper:** what to add when the interviewer pushes.
- **From my projects:** a real example with real numbers, from [rag-docs](https://github.com/Vsaraf334/rag-docs) and [incident-agent](https://github.com/Vsaraf334/incident-agent).
- **Follow-ups:** what they're likely to ask next.

Revise by reading only the short answers first, then saying the rest out loud from memory.

| Area | File | Questions |
| --- | --- | --- |
| A | [LLM fundamentals](A-llm-fundamentals.md) | 10 |
| B | Prompting and structured output | 8 (to do) |
| C | RAG | 15 (to do) |
| D | Agents and tool use | 12 (to do) |
| E | MCP and frameworks | 8 (to do) |
| F | Evaluation | 12 (to do) |
| G | Cost and latency | 8 (to do) |
| H | Production and reliability | 10 (to do) |
| I | Security and safety | 8 (to do) |
| J | Fine-tuning and model choices | 5 (to do) |
| K | My projects and behavioural | 6 (to do) |
| L | System design | 5 (to do) |

**Numbers to have ready** (from the projects):

- rag-docs pipeline: 19/20 answers correct, 4/4 unanswerable questions declined; the agent version 20/20 at about 2.3× the tokens.
- incident-agent: 15/15 scenarios and 45/45 runs with Sonnet 5.5; Haiku 5.5 found the right cause in 45/45 runs but could not file in 7 (15 failed tool calls from leaked markup), and used 48% more tokens per run (Sonnet used 32% fewer).
- Same code, two Haiku evals: 42/45 and 37/45 runs. That is the noise level.
- LangGraph rebuild: same scores as the hand-written loop (45/45).
