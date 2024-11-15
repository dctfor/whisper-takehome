from models import ChatMessage, ChatHistory
import dspy
from lms.together import Together

from modules.optimizer import prepare_training_data, create_and_compile_chatbot, sanitize_input, get_last_response

lm = Together(
    model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    temperature=0.5,
    max_tokens=1000,
    top_p=0.7,
    top_k=50,
    repetition_penalty=1.2,
    stop=["<|eot_id|>", "<|eom_id|>", "\n\n---\n\n", "\n\n---", "---", "\n---"],
    # stop=["\n", "\n\n"],
)

dspy.settings.configure(lm=lm)

chat_history = ChatHistory()

# Prepare the training data
training_data = prepare_training_data('training_data/conversations.json')

# Create and compile the chatbot
compiled_chatbot = create_and_compile_chatbot(training_data, k=3)

while True:
    # Get user input
    user_input = input("\033[94m> You:\033[0m ")
    # Append user input to chat history
    chat_history.messages.append(
        ChatMessage(
            from_creator=False,
            content=sanitize_input(user_input),
        ),
    )
    # Send request to endpoint
    response = compiled_chatbot(context=chat_history.model_dump_json())
    # Validate if had more reasoning and has several responses - long conversations case
    if "\n" in response:
        response = get_last_response(response)
    # Append response to chat history
    chat_history.messages.append(
        ChatMessage(
            from_creator=True,
            content=response,
        ),
    )
    # Print response with extra line break and color green for the text in some terminals
    print("\n\033[92m> Response:", response,'\033[0m\n')
    # uncomment this line to see the 
    # lm.inspect_history(n=1)
