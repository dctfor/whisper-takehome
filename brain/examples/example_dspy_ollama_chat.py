import dspy
import json
import ftfy
import re
from dspy.teleprompt import KNNFewShot
from dspy import Example
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional


class ChatMessage(BaseModel):
    from_creator: bool
    content: str
    def __str__(self):
        role = "YOU" if self.from_creator else "THE FAN"
        message = role + ": " + self.content
        return message

class ChatHistory(BaseModel):
    messages: List[ChatMessage] = []
    def __str__(self):
        messages = []
        for i, message in enumerate(self.messages):
            message_str = str(message)
            messages.append(message_str)
        return "\n".join(messages)
    def model_dump_json(self, **kwargs):
        return str(self)


class TopicFilter:
    """Filter for managing topic restrictions from the LLM."""
    def __init__(self):
        # short list of plataforms
        self.restricted_platforms = [
            "facebook", "instagram", "twitter", "x.com", "tiktok", 
            "snapchat", "linkedin", "pinterest", "reddit",
            "telegram", "discord", "twitch", "whatsapp",
        ]
        self.onlyfans_alternatives = [
            "Fansly", "Fancentro", "MYM.fans", "JustFor.Fans",
            "FanTime", "Fanvue", "OkFans",
            "iFans", "Modelhub", "TipSnaps", "Vanywhere",
            "Directs", "Fanhouse", "Spore", "Ask.fm",
        ]
        # Patterns for in-person meeting suggestions
        self.meeting_patterns = [
            r"meet up", r"meet in person", r"get together", 
            r"meet you", r"see you there", r"join me", 
            r"come to", r"attend my", r"in person",
        ]
    def contains_restricted_content(self, text: str) -> tuple[bool, str]:
        """
        Check if text contains any restricted content.
        Returns (has_restricted_content, reason)
        """
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        # Check for social media platforms
        for platform in self.restricted_platforms:
            if platform in text_lower:
                return True, f"Contains reference to {platform}"
        # Check for meeting suggestions
        for pattern in self.meeting_patterns:
            if re.search(pattern, text_lower):
                return True, "Contains suggestion for in-person meeting"
        return False, ""

class ChatResponse(dspy.Signature):
    """Signature for generating chat responses with content filtering."""
    context = dspy.InputField(desc="Previous messages in the conversation currently going on")
    response = dspy.OutputField(desc="""
        Generate only one short response in 1st person of 1 or 2 sentences that matches the creator's style, no extra comments, reasoning nor explanations needed.
        """)

class StyleOptimizedChatbot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(ChatResponse)
        self.topic_filter = TopicFilter()
    def forward(self, context: ChatHistory) -> str:
        """Generate a filtered response given the conversation context."""
        max_attempts = 3
        for attempt in range(max_attempts):
            pred = self.generator(context=context)
            response = pred.response
            # Check if response contains restricted content
            has_restricted, reason = self.topic_filter.contains_restricted_content(response)
            if not has_restricted:
                return response
            # If we're on the last attempt and still getting restricted content,
            # generate a safe fallback response
            if attempt == max_attempts - 1:
                return self._generate_safe_response()
        return self._generate_safe_response()
    def _generate_safe_response(self) -> str:
        """Generate a safe, generic response when filtering fails."""
        # ... yet to improve
        safe_responses = [
            "That's a beautiful perspective. Keep exploring what resonates with you.",
            "Your unique way of seeing things is what makes you special.",
            "It's wonderful how you find meaning in these moments.",
            "That's such an introspective way to look at it.",
            "Your authenticity shines through in how you approach this.",
            "I appreciate how you reflect on these experiences.",
            "Your thoughts are truly inspiring.",
            "It's amazing how you find depth in everyday moments.",
            "Your perspective adds so much value to the conversation.",
            "Thank you for sharing your unique insights."
        ]
        return safe_responses[hash(str(dspy.settings.lm)) % len(safe_responses)]

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


lm = dspy.OllamaLocal(
    model="llama3.1",
    max_tokens=8000,
    timeout_s=480
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

