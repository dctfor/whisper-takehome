So instead of doing a basic off-the-shelf take home which is probably now easily solved using chatgpt or something, I thought it'd be better to have it be more custom-fit to the problem Whisper is solving.

# Project Overview

This takehome is basically a super dumbed down version of the product Whisper makes. It uses DSPy, which is a tool useful for making LLM-based applications. It has some pretty interesting abstractions which I like and have found convenient for tinkering and building in the space we're working in.

The takehome already contains a somewhat functioning chatbot. The first step is to get the chatbot to run and talk to it. If you're using VS code, to do this, add the following vs code configuration and press play:

{
    "name": "Python: local_chat",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/brain/chat_interface.py",
    "console": "integratedTerminal",
    "env": {
        "TOGETHER_API_KEY": ${api key here. You can make a free account at together.ai to get an api key},
    }
},

Then you can chat with the chatbot. Let me know if you have issues doing this.

# Goals

This chatbot as it stands is pretty basic. For one, we want it to sound more like our client. We have already collected a few fake example conversations in training_data/conversations.json. We also want to improve it more generally.

1. **Improve Client Personality Emulation**  
   Use DSPy’s KNNFewShot optimizer (https://dspy.ai/learn/optimization/optimizers/) to make the chatbot’s responses reflect our client’s voice more authentically, based on examples in `conversations.json`. (Done)

2. **Incorporate Context Awareness**  
   Introduce context awareness in a way that makes the chatbot more responsive to the timing and circumstances of each interaction. Examples might include awareness of the current time or the duration of a conversation. (Done)

3. **Topic Filtering**  
   Ensure the chatbot avoids discussing specific topics that may not be suitable. For this exercise, keep responses free of mentions of social media platforms (except OnlyFans <- LOL) and interactions suggesting in-person meetings with fans. (Partially done - :/)

4. **Further Product Enhancements**  
   Identify and implement an additional enhancement that you believe would improve the product experience. (WIP)

The first goal is probably the hardest, but I want it done first and it will be what I look at closest. 

The things I'm looking for are:
 1. Can you quickly learn a new framework/new technology 
 2. How do you think about product improvements 
 3. How do you think about implementing these product improvements using dspy.

I'm not holding your hand much on this take-home on purpose as I'd like to feel confident you can take on these challenges independently.

**Note**: Avoid spending time on extensive prompt engineering. At Whisper, we value modular and maintainable code, and we prefer optimizations within DSPy itself rather than large, static prompts. Also, to see the actual prompts dspy is generating, uncomment the lm.inspect_history(n=1) line in `chat_interface.py`.

Please leave comments or notes on your thought process and what you built in a separate README file for me to take a look at.

I am expecting you to work on this for 2-3 hours. You can work on it longer if you'd like, just let me know how long you end up working on it.

Good luck!


# Feedback

⏱️ **Total Time Spent:**  
Tbh, total time spent here 7.5hrs due my learning curve, experimenting different cases and setting up/fixing a local testing.

IMPORTANT . Due a version issue or something I did in my testings, I was asked by the errors to directly set the api_key on the client var (Together object) @ brain/lms/together.py Line ~10 or 9

## Some Thoughts

🧪 **Testing:**  
* Tested with Ollama/llama3.1 8B (modded to 32k ctx), as part of my budget-friendly focus. Yet can't keep with uncensored talks.

💡 **Model Considerations:**  
* My POV is, using a 405B might be overkill for a simple chatbot unless it explicitly is the best option so far for preventing hallucinations or to skip the censorship and also like its responses. In any case, there are more models with lower parameters that allow uncensored talks which would make easier to allow topics related with onlyfans or related keeping up the profesionalism.

## TODOs:

🛠️ **Improvements:**
* Would also implement a sentiment analyzer to enable better context handling so it can be a different chatbot with different prompts for specifying prompt-behavioral pipelines. The basic version is a positive/neutral/negative detector but there are more fancy analyzer pipelines which can tell emotions. 

* Would go for an uncensored and smaller llm depending on the ethics guidelines.

* Could use [Groq](https://groq.com/) or [Cerebras](https://inference.cerebras.ai/) for speed in case the smaller LLM is allowed

🔒 **Security:**
* Would look for more security breaches on LLM/SLM chatbots, malicious prompts and stuff, currently just a minor sanitization.

🚧 **Fallbacks:**
* Implement more fallbacks/failsafes pipelines with edge cases so doesn't feel that weird and make it more human, maybe with a complex pipeline with the LLM for a fallback with a second chat response model.

⚙️ **Settings:**
* Can there be a setting to allow/forbid certain categories of talk besides the short training dataset? This for a ethical custom setup for specific topics or guidelines. Also can be set the plataform the bot will work with.

😊 **Experience:**
* I had a happy learning time doing this challenge/takehome :D

(\_/)
( n.n)b
c( ")( ")
Whatever happens,
thanks for this experience!