"""
Token Translator Core Module
Pure conversion functions without file I/O or command line interface
Easy to embed in other Python projects
"""

import requests
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class DatabasePriority(Enum):
    """Priority of token databases"""
    SLOVA = 0
    RUSSIAN = 1
    NAMES = 2

@dataclass
class TokenInfo:
    """Information about a single token"""
    id: str
    source: Optional[DatabasePriority]
    is_full: bool

@dataclass
class TranslationResult:
    """Result of translation with statistics"""
    content: str
    total_tokens: int = 0
    slova_tokens: int = 0
    russian_tokens: int = 0
    names_tokens: int = 0
    partial_tokens: int = 0

class TokenConverter:
    """
    Core token conversion class
    Can be easily embedded in other projects
    """
    
    # Database URLs - can be customized
    DATABASE_URLS = {
        "slova.json": "https://raw.githubusercontent.com/scream-dev/Scream-Dev.ru/refs/heads/main/cdn/slova.json",
        "russian.json": "https://dl.dropboxusercontent.com/scl/fi/l7u0na2cp99btx3etxskx/russian.json?rlkey=wgi95f6vq32cpdt48nmzajxu0&st=28buk26k&dl=0",
        "russian-names.json": "https://dl.dropboxusercontent.com/scl/fi/cfogog0iaoacjw77qxr3p/russian_surnames.json?rlkey=splkwdgmrncenwekwocs1b2jw&st=sfwcna67&dl=0"
    }
    
    def __init__(self, 
                 token_map_slova: Optional[Dict[str, str]] = None,
                 token_map_russian: Optional[Dict[str, str]] = None,
                 token_map_names: Optional[Dict[str, str]] = None,
                 auto_load: bool = True):
        """
        Initialize token converter
        
        Args:
            token_map_slova: Pre-loaded slova tokens (optional)
            token_map_russian: Pre-loaded russian tokens (optional)
            token_map_names: Pre-loaded names tokens (optional)
            auto_load: Automatically load tokens from URLs if not provided
        """
        self.token_map_slova = token_map_slova or {}
        self.token_map_russian = token_map_russian or {}
        self.token_map_names = token_map_names or {}
        
        if auto_load and not any([token_map_slova, token_map_russian, token_map_names]):
            self.load_all_tokens()
    
    @staticmethod
    def download_json(url: str) -> Optional[Dict]:
        """Download JSON from URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None
    
    def load_all_tokens(self) -> bool:
        """Load all token databases from URLs"""
        # Load slova.json
        slova_data = self.download_json(self.DATABASE_URLS["slova.json"])
        if slova_data and "token_to_id" in slova_data:
            self.token_map_slova = {k: str(v) for k, v in slova_data["token_to_id"].items()}
        
        # Load russian.json
        russian_data = self.download_json(self.DATABASE_URLS["russian.json"])
        if russian_data and "token_to_id" in russian_data:
            self.token_map_russian = {k: str(v) for k, v in russian_data["token_to_id"].items()}
        
        # Load russian-names.json
        names_data = self.download_json(self.DATABASE_URLS["russian-names.json"])
        if names_data and "token_to_id" in names_data:
            self.token_map_names = {k: str(v) for k, v in names_data["token_to_id"].items()}
        
        return any([self.token_map_slova, self.token_map_russian, self.token_map_names])
    
    def set_token_maps(self, 
                       slova: Dict[str, str] = None,
                       russian: Dict[str, str] = None,
                       names: Dict[str, str] = None):
        """Manually set token maps"""
        if slova:
            self.token_map_slova = slova
        if russian:
            self.token_map_russian = russian
        if names:
            self.token_map_names = names
    
    def get_token_id(self, word: str) -> Optional[TokenInfo]:
        """Find full token match for a word"""
        lower_word = word.lower().strip()
        
        if lower_word in self.token_map_slova:
            return TokenInfo(self.token_map_slova[lower_word], DatabasePriority.SLOVA, True)
        if lower_word in self.token_map_russian:
            return TokenInfo(self.token_map_russian[lower_word], DatabasePriority.RUSSIAN, True)
        if lower_word in self.token_map_names:
            return TokenInfo(self.token_map_names[lower_word], DatabasePriority.NAMES, True)
        
        return None
    
    def find_longest_token(self, word: str, start_index: int = 0) -> Optional[Tuple[str, str, DatabasePriority, int, int]]:
        """Find the longest token starting from given position"""
        search_word = word.lower()
        best_match = None
        best_length = 0
        
        # Search in slova.json
        for token, token_id in self.token_map_slova.items():
            if search_word.startswith(token, start_index) and len(token) > best_length:
                best_match = (token, token_id, DatabasePriority.SLOVA, start_index, start_index + len(token))
                best_length = len(token)
        
        # Search in russian.json
        for token, token_id in self.token_map_russian.items():
            if search_word.startswith(token, start_index) and len(token) > best_length:
                best_match = (token, token_id, DatabasePriority.RUSSIAN, start_index, start_index + len(token))
                best_length = len(token)
        
        # Search in names.json
        for token, token_id in self.token_map_names.items():
            if search_word.startswith(token, start_index) and len(token) > best_length:
                best_match = (token, token_id, DatabasePriority.NAMES, start_index, start_index + len(token))
                best_length = len(token)
        
        return best_match
    
    def tokenize_word(self, word: str) -> List[TokenInfo]:
        """Tokenize a single word into tokens"""
        result = []
        
        if not word:
            return result
        
        lower_word = word.lower().strip()
        full_match = self.get_token_id(lower_word)
        
        if full_match:
            return [full_match]
        
        current_index = 0
        while current_index < len(lower_word):
            match = self.find_longest_token(lower_word, current_index)
            
            if match:
                token, token_id, source, start, end = match
                result.append(TokenInfo(token_id, source, True))
                current_index = end
                
                # Add separator "-" if there is continuation
                if current_index < len(lower_word) and result:
                    result.append(TokenInfo("-", None, False))
            else:
                # Partial token for single character
                partial_char = lower_word[current_index]
                result.append(TokenInfo(partial_char, None, False))
                current_index += 1
                
                if current_index < len(lower_word):
                    result.append(TokenInfo("-", None, False))
        
        return result
    
    def text_to_tokens(self, text: str) -> TranslationResult:
        """
        Convert text to tokens
        
        Args:
            text: Input text string
            
        Returns:
            TranslationResult with tokens and statistics
        """
        if not text:
            return TranslationResult("")
        
        words = text.split()
        result_parts = []
        
        result = TranslationResult("")
        
        for word_idx, word in enumerate(words):
            tokenized = self.tokenize_word(word)
            
            if word_idx > 0 and tokenized:
                result_parts.append("+")
            
            for token_idx, token in enumerate(tokenized):
                if token.source is None and token.id == "-":
                    result_parts.append("-")
                elif token.source is None and token.id != "-":
                    result_parts.append(token.id)
                    result.total_tokens += 1
                    result.partial_tokens += 1
                else:
                    if token_idx > 0 and result_parts[-1] != "-":
                        if result_parts and result_parts[-1] not in ["+", "-"]:
                            result_parts.append("-")
                    result_parts.append(token.id)
                    result.total_tokens += 1
                    
                    if token.source == DatabasePriority.SLOVA:
                        result.slova_tokens += 1
                    elif token.source == DatabasePriority.RUSSIAN:
                        result.russian_tokens += 1
                    elif token.source == DatabasePriority.NAMES:
                        result.names_tokens += 1
        
        result.content = "".join(result_parts)
        return result
    
    def tokens_to_text(self, tokens: str) -> TranslationResult:
        """
        Convert tokens back to text
        
        Args:
            tokens: Token string with + and - separators
            
        Returns:
            TranslationResult with text and statistics
        """
        if not tokens:
            return TranslationResult("")
        
        # Create reverse maps
        reverse_slova = {v: k for k, v in self.token_map_slova.items()}
        reverse_russian = {v: k for k, v in self.token_map_russian.items()}
        reverse_names = {v: k for k, v in self.token_map_names.items()}
        
        result_parts = []
        current_word = []
        
        result = TranslationResult("")
        
        current_token = ""
        for char in tokens:
            if char in ['+', '-']:
                if current_token:
                    result.total_tokens += 1
                    
                    # Check if it's a number (token ID) or character
                    if current_token.isdigit():
                        # Search in reverse maps
                        if current_token in reverse_slova:
                            current_word.append(reverse_slova[current_token])
                            result.slova_tokens += 1
                        elif current_token in reverse_russian:
                            current_word.append(reverse_russian[current_token])
                            result.russian_tokens += 1
                        elif current_token in reverse_names:
                            current_word.append(reverse_names[current_token])
                            result.names_tokens += 1
                    else:
                        # Partial token (single character)
                        current_word.append(current_token)
                        result.partial_tokens += 1
                    
                    current_token = ""
                
                if char == '+':
                    if current_word:
                        result_parts.append("".join(current_word))
                        current_word = []
                    if result_parts and result_parts[-1] != " ":
                        result_parts.append(" ")
                # '-' means continue word, do nothing
            else:
                current_token += char
        
        # Process last token
        if current_token:
            result.total_tokens += 1
            if current_token.isdigit():
                if current_token in reverse_slova:
                    current_word.append(reverse_slova[current_token])
                    result.slova_tokens += 1
                elif current_token in reverse_russian:
                    current_word.append(reverse_russian[current_token])
                    result.russian_tokens += 1
                elif current_token in reverse_names:
                    current_word.append(reverse_names[current_token])
                    result.names_tokens += 1
            else:
                current_word.append(current_token)
                result.partial_tokens += 1
        
        if current_word:
            result_parts.append("".join(current_word))
        
        result.content = "".join(result_parts).strip()
        return result
    
    def get_stats_summary(self, result: TranslationResult) -> Dict[str, Any]:
        """Get statistics summary as dictionary"""
        return {
            'total_tokens': result.total_tokens,
            'slova_tokens': result.slova_tokens,
            'russian_tokens': result.russian_tokens,
            'names_tokens': result.names_tokens,
            'partial_tokens': result.partial_tokens,
            'content_length': len(result.content)
        }
    
    def calculate_compression(self, original_text: str, tokenized_text: str) -> float:
        """
        Calculate compression percentage
        
        Returns:
            Positive percentage if compressed (smaller),
            Negative if expanded (larger)
        """
        if not original_text:
            return 0.0
        return ((len(original_text) - len(tokenized_text)) / len(original_text)) * 100.0


# Convenience functions for easy embedding
def create_converter(preloaded_tokens: Dict[str, Dict[str, str]] = None) -> TokenConverter:
    """
    Create a TokenConverter instance with optional pre-loaded tokens
    
    Args:
        preloaded_tokens: Dictionary with 'slova', 'russian', 'names' keys
        
    Returns:
        TokenConverter instance
    """
    if preloaded_tokens:
        return TokenConverter(
            token_map_slova=preloaded_tokens.get('slova'),
            token_map_russian=preloaded_tokens.get('russian'),
            token_map_names=preloaded_tokens.get('names'),
            auto_load=False
        )
    return TokenConverter()

def text_to_tokens_simple(text: str, converter: TokenConverter = None) -> str:
    """
    Simple text to tokens conversion
    
    Args:
        text: Input text
        converter: Optional TokenConverter instance
        
    Returns:
        Token string
    """
    if converter is None:
        converter = TokenConverter()
    return converter.text_to_tokens(text).content

def tokens_to_text_simple(tokens: str, converter: TokenConverter = None) -> str:
    """
    Simple tokens to text conversion
    
    Args:
        tokens: Token string
        converter: Optional TokenConverter instance
        
    Returns:
        Text string
    """
    if converter is None:
        converter = TokenConverter()
    return converter.tokens_to_text(tokens).content

# Example of embedding in another project
"""
# Example usage in another Python file:

from token_converter_core import TokenConverter, text_to_tokens_simple

# Option 1: Simple usage
tokens = text_to_tokens_simple("Привет мир")
print(tokens)

# Option 2: Full control
converter = TokenConverter()
result = converter.text_to_tokens("Hello world")
print(f"Tokens: {result.content}")
print(f"Stats: {converter.get_stats_summary(result)}")

# Option 3: With pre-loaded tokens
my_tokens = {
    'slova': {'привет': '100', 'мир': '101'},
    'russian': {'hello': '200', 'world': '201'}
}
converter = TokenConverter(
    token_map_slova=my_tokens['slova'],
    token_map_russian=my_tokens['russian'],
    auto_load=False
)
"""