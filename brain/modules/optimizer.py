import json
import ftfy
import re
from dspy.teleprompt import KNNFewShot
from dspy import Example
from modules.chatter import StyleOptimizedChatbot

def prepare_training_data(json_file_path: str) -> list:
    """Prepare training data from JSON file with improved conversation context handling."""
    with open(json_file_path, encoding='utf-8') as f:
        conversations = json.load(f)
    training_data = []
    for conversation in conversations:
        messages = conversation['chat_history']['messages']
        context = "\n".join([
            f"{'User' if not msg['from_creator'] else 'Creator'}: {ftfy.fix_text(msg['content'])}"
            for msg in messages
        ])
        output = ftfy.fix_text(conversation['output'])
        example = Example(
            context=context,
            response=output
        ).with_inputs('context')
        training_data.append(example)
    return training_data

def sanitize_input(input_text):
    # Remove all characters except letters, numbers, and basic punctuation
    sanitized = re.sub(r"[^a-zA-Z0-9\s.,!?']", '', input_text)
    return sanitized

def create_and_compile_chatbot(training_data: list, k: int = 5) -> StyleOptimizedChatbot:
    """Create and compile the chatbot with KNNFewShot optimizer."""
    optimizer = KNNFewShot(k=k, trainset=training_data)
    compiled_chatbot = optimizer.compile(StyleOptimizedChatbot(), trainset=training_data)
    return compiled_chatbot

def get_last_response(llm_output):
    """
    Extracts the last 'Response' from the LLM output.
    Just in case we have more responses than needed due an hallucination.
    """
    # Regular expression pattern to find all occurrences of 'Response:'
    pattern = r'Response:(.*?)(?=\n[A-Z][^:]*:|\Z)'
    responses = re.findall(pattern, llm_output, re.DOTALL)
    if responses:
        # Get the content of the last 'Response:'
        last_response = responses[-1].strip()
        return last_response
    else:
        return "..."
