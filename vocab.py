"""Vocabulary builder and tokenizer for synthetic object-property sentences."""

import json
from typing import Dict, List, Tuple
from pathlib import Path


class Vocabulary:
    """Manages token-to-ID mappings for the synthetic language."""

    # Special tokens
    PAD_TOKEN = "[PAD]"
    CLS_TOKEN = "[CLS]"
    UNK_TOKEN = "[UNK]"

    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Initialize special tokens with fixed IDs."""
        special_tokens = [self.PAD_TOKEN, self.CLS_TOKEN, self.UNK_TOKEN]
        for i, token in enumerate(special_tokens):
            self.token2id[token] = i
            self.id2token[i] = token

    def add_token(self, token: str) -> int:
        """Add a token to vocabulary if not present. Returns token ID."""
        if token not in self.token2id:
            token_id = len(self.token2id)
            self.token2id[token] = token_id
            self.id2token[token_id] = token
        return self.token2id[token]

    def add_tokens(self, tokens: List[str]):
        """Add multiple tokens to vocabulary."""
        for token in tokens:
            self.add_token(token)

    def get_id(self, token: str) -> int:
        """Get ID for token, returns UNK_TOKEN ID if not found."""
        return self.token2id.get(token, self.token2id[self.UNK_TOKEN])

    def get_token(self, token_id: int) -> str:
        """Get token for ID."""
        return self.id2token.get(token_id, self.UNK_TOKEN)

    def __len__(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return self.token2id[self.PAD_TOKEN]

    @property
    def cls_id(self) -> int:
        return self.token2id[self.CLS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token2id[self.UNK_TOKEN]

    def save(self, path: str):
        """Save vocabulary to JSON file."""
        with open(path, 'w') as f:
            json.dump({
                'token2id': self.token2id,
                'id2token': {int(k): v for k, v in self.id2token.items()}
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load vocabulary from JSON file."""
        vocab = cls()
        with open(path, 'r') as f:
            data = json.load(f)
        vocab.token2id = data['token2id']
        vocab.id2token = {int(k): v for k, v in data['id2token'].items()}
        return vocab


def build_vocab_from_config(config_path: str) -> Vocabulary:
    """Build vocabulary from sentence generation config file.

    Args:
        config_path: Path to JSON config with colors, sizes, shapes, rels.

    Returns:
        Vocabulary object with all tokens from config.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    vocab = Vocabulary()

    # Add all tokens from config
    vocab.add_tokens(config.get('colors', []))
    vocab.add_tokens(config.get('sizes', []))
    vocab.add_tokens(config.get('shapes', []))
    vocab.add_tokens(config.get('rels', []))

    # Add common glue words that might appear
    # (Based on the spec, relationships are used directly, but "and" appears)
    # The "and" is already in rels in sentence_config.json

    return vocab


class Tokenizer:
    """Tokenizes sentences into token IDs with attention masks."""

    def __init__(self, vocab: Vocabulary, max_len: int = 12):
        self.vocab = vocab
        self.max_len = max_len

    def encode(self, sentence: str) -> Tuple[List[int], List[int]]:
        """Tokenize sentence to token IDs and attention mask.

        Args:
            sentence: Input sentence string.

        Returns:
            Tuple of (token_ids, attention_mask).
            - token_ids: List of length max_len with [CLS] prepended.
            - attention_mask: 1 for real tokens, 0 for PAD.
        """
        # Split on whitespace
        tokens = sentence.strip().split()

        # Prepend [CLS] token
        tokens = [self.vocab.CLS_TOKEN] + tokens

        # Convert to IDs
        token_ids = [self.vocab.get_id(token) for token in tokens]

        # Compute actual length (before padding)
        actual_len = len(token_ids)

        # Truncate if too long
        if actual_len > self.max_len:
            token_ids = token_ids[:self.max_len]
            actual_len = self.max_len

        # Create attention mask (1 for real tokens)
        attention_mask = [1] * actual_len

        # Pad to max_len
        padding_len = self.max_len - actual_len
        token_ids += [self.vocab.pad_id] * padding_len
        attention_mask += [0] * padding_len

        return token_ids, attention_mask

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to sentence.

        Args:
            token_ids: List of token IDs.
            skip_special: If True, skip special tokens like [PAD], [CLS].

        Returns:
            Decoded sentence string.
        """
        special_tokens = {self.vocab.PAD_TOKEN, self.vocab.CLS_TOKEN}
        tokens = []

        for token_id in token_ids:
            token = self.vocab.get_token(token_id)
            if skip_special and token in special_tokens:
                continue
            tokens.append(token)

        return " ".join(tokens)
