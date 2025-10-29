<div align="center">
 
![logo](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant/blob/main/assets/logo_rounded.png)

<h1 align="center"><strong>🤖 Agent Details :<h6 align="center">All implemented agents have been detailed below</h6></strong></h1>

</div>

---
 
## 📚 Table of Contents
- [Human-in-the-loop Validation Agent](#human-in-the-loop)

---

## 📌 Human-in-the-loop validation of Medical Computer Vision Diagnosis Agents' Outputs <a name="human-in-the-loop"></a>

In `agent_decision.py`:

1. Interrupt the workflow when human validation is needed
2. Store the interrupted state in memory
3. Add endpoints to expose pending validations and submit validation decisions
4. Resume the workflow after the human has provided feedback

On frontend:

1. Check if a response needs validation (needs_validation flag)
2. If so, show a validation interface to the human reviewer
3. Send the validation decision back through the /validate endpoint
4. Continue the conversation

Implemented a complete human-in-the-loop validation system using LangGraph's NodeInterrupt functionality, integrated with the backend and frontend.

---