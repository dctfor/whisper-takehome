import dspy

import re
from models import ChatHistory

class TopicFilter:
    """Filter for managing topic restrictions from the LLM."""
    def __init__(self):
        # short list of plataforms
        self.restricted_platforms = [
            "facebook", "instagram", "twitter", "x.com", "tiktok", 
            "snapchat", "linkedin", "pinterest", "reddit",
            "telegram", "discord", "twitch", "whatsapp",
        ]

        # yet to be implemented
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
        Take the role of a Onlyfans content creator.
        Generate 1st person short response of 1 or 2 sentences that matches the creator's style, no extra comments nor explanations needed.
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