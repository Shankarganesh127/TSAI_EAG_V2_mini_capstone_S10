
'''
input query dissected and given to perception. 

once perception is clear, 

there should be decision should be included with planning, required actions based on existing mcp tools (web search, document search, previous conversation history results, in future can be added in future so need to ) , memory content extract functions, 

START
  ↓
INPUT_RECEIVED
  ↓
PERCEPTION
  ↓
CONTEXT_RETRIEVAL
  ↓
DECISION
  ↓
PLANNING
  ↓
ACTION
  ↓
OBSERVATION
  ↓
VALIDATION
  ↓
REFLECTION / REPLAN
  ↓
OUTPUT
  ↓
END


Perception
   ↓
Context Retrieval
   ↓
Decision
   ↓
Planning
   ↓
Action
   ↓
Observation
   ↓
Validation
   ↓
Reflection/Replan if needed
   ↓
Output

agent_framework/

├── core/
│   ├── state.py
│   ├── context.py
│   ├── result.py
│   └── exceptions.py
│
├── stages/
│   ├── base_stage.py
│   ├── perception.py
│   ├── decision.py
│   ├── planning.py
│   ├── action.py
│   ├── validation.py
│   └── reflection.py
│
├── orchestrator/
│   ├── fsm.py
│   └── orchestrator.py

So create a simple base classes. for each stages, I will organize the stages flow based on LLM response itself. 
'''

import asyncio

from agents.base_agent import BaseAgent


async def main():

    agent = BaseAgent()

    result = await agent.run(
        "Find current F1 standings"
    )

    print(result)


asyncio.run(main())