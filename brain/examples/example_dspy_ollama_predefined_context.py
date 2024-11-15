import os
import json
import ftfy
import dspy
import re
from dspy.teleprompt import KNNFewShot
from dspy import Example, InputField, OutputField, Signature, Module, Predict, Prediction

# Initialize Ollama
lm = dspy.OllamaLocal(
    model="llama3.1",
    max_tokens=8000,
    timeout_s=480
)
dspy.settings.configure(lm=lm)

class TopicFilter:
    """Filter for managing topic restrictions from the LLM."""
    def __init__(self):
        # short list of plataforms
        self.restricted_platforms = [
            "facebook", "instagram", "twitter", "x.com", "tiktok", 
            "snapchat", "linkedin", "pinterest", "reddit", "whatsapp",
            "telegram", "discord", "twitch"
        ]
        # Patterns for in-person meeting suggestions
        self.meeting_patterns = [
            r"meet up", r"meet in person", r"get together", 
            r"meet you", r"see you there", r"join me", 
            r"come to", r"attend my", r"in person"
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
        Generate 1st person short response of 1 or 2 sentences that matches the creator's style, no extra comments nor explanations needed.
        """)

class StyleOptimizedChatbot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(ChatResponse)
        self.topic_filter = TopicFilter()
    
    def forward(self, context: str) -> str:
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

def test_chatbot(chatbot: StyleOptimizedChatbot, test_contexts: list):
    """Test the chatbot with some example contexts."""
    for context in test_contexts:
        response = chatbot(context=sanitize_input(context))
        print(f"\n> > > Context:\n{context}\n> > > Response: {response}\n")
        print("-" * 50)

def main():
    # Prepare the training data
    training_data = prepare_training_data('training_data/conversations.json')
    # Create and compile the chatbot
    compiled_chatbot = create_and_compile_chatbot(training_data, k=3)
    # Test contexts including some that might trigger filters
    test_contexts = [
        """
        User: I spend hours organizing my books by color instead of actually reading them...
        Creator: Sometimes the act of organizing can be its own form of meditation and connection.
        User: I never thought about it that way. Maybe it's not just procrastination?
        """,
        """
        User: I'd love to show you my art collection! Maybe we could meet up sometime?
        Creator: Art collections are such personal journeys. Each piece tells a story.
        User: Yes! I can share some photos on Instagram if you'd like to see them.
        """,
        """
        User: Do you ever share your daily routines on social media?
        Creator: I try to keep things authentic and meaningful.
        User: Maybe I could join one of your workshops sometime?
        """,
        """
        User: I feel like I'm collecting moments but not really experiencing them.
        Creator: That's an interesting perspective. Maybe we need both - experiencing and reflecting.
        User: Like creating a personal museum of memories?
        """,
        """
        User: I've been binge-watching TikTok videos lately. It's hard to stop!
        Creator: Those quick bursts of creativity can be addictive! Have you thought of creating your own?
        User: Maybe... but I'm not sure if I have anything interesting to share.
        """,
        """
        User: I want to start journaling, but I never know what to write.
        Creator: Sometimes starting with just a single thought or a question can open the floodgates.
        User: Good idea! I could use it as a way to reflect on my day.
        """,
        """
        User: I've been thinking about going vegan, but I don't know where to start.
        Creator: It's all about small steps. Maybe start with one plant-based meal a day?
        User: That seems manageable. Any recipes you'd recommend?
        """,
        """
        User: I feel like Instagram is making me anxious about not doing enough with my life.
        Creator: Social media can create unrealistic expectations. Remember, it's just a highlight reel.
        User: Yeah, I guess I need to take a break and focus on what makes me happy.
        """,
        """
        User: I've been into vintage photography lately. Do you shoot on film?
        Creator: Absolutely! There's something magical about the anticipation of developing film.
        User: I love that! Any tips for someone just starting out?
        """,
        """
        User: I tried meditating, but my mind just keeps wandering.
        Creator: That's perfectly normal. It's more about bringing your focus back each time it drifts.
        User: I guess I need more practice. Do you use any specific techniques?
        """,
        """
        User: Twitter is so chaotic, but I can't help checking it all the time.
        Creator: It's easy to get pulled in! Sometimes setting a daily limit can help.
        User: I might try that... or maybe mute some of the noisier accounts.
        """,
        """
        User: I've been learning to code, but sometimes it feels overwhelming.
        Creator: It can be at first, but every little breakthrough is so rewarding.
        User: I suppose I just need to celebrate the small wins along the way.
        """,
        """
        User: I just started a YouTube channel, but I'm worried no one will watch my videos.
        Creator: Everyone starts somewhere. Focus on creating content you're passionate about.
        User: You're right. I'll just have fun with it for now.
        """,
        """
        User: I feel like I'm always chasing after the next thing without appreciating what I have.
        Creator: It's easy to get caught up in that mindset. Sometimes slowing down brings more clarity.
        User: I think I need to take a break and just breathe for a while.
        """
    ]
    # Test the chatbot
    test_chatbot(compiled_chatbot, test_contexts)


if __name__ == "__main__":
    main()

