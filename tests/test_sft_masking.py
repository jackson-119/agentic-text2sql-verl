"""Synthetic-template contract tests, not real-tokenizer replay."""
import copy
import unittest
from agentic_text2sql.sft.masking import clean_messages, tokenize_with_mask


class TinyTokenizer:
    roles = {"system": 10, "user": 11, "assistant": 12, "tool": 13}

    def apply_chat_template(
        self, messages, tools, tokenize, add_generation_prompt
    ):
        assert tokenize is True
        assert all("trainable" not in message for message in messages)
        result = []
        for message in messages:
            result += [self.roles[message["role"]]]
            result += [100 + ord(c) for c in message.get("content", "")]
            result += [2]
        if add_generation_prompt:
            result += [12]
        return result


class BadGenerationPrefix(TinyTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        result = super().apply_chat_template(*args, **kwargs)
        if kwargs["add_generation_prompt"]:
            result[-1] = 99
        return result


class BadConversationPrefix(TinyTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        result = super().apply_chat_template(messages, **kwargs)
        if len(messages) >= 3:
            result[0] = 99
        return result


def messages():
    return [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "bc"},
        {"role": "tool", "content": "d"},
        {"role": "assistant", "content": "e"},
    ]


class SFTMaskingTests(unittest.TestCase):
    def test_assistant_only_and_headers_excluded(self):
        item = tokenize_with_mask(TinyTokenizer(), messages(), [], 100)
        self.assertEqual(
            item["labels"],
            [-100, -100, -100, -100, 198, 199, 2,
             -100, -100, -100, -100, 201, 2],
        )
        self.assertEqual(item["attention_mask"], [1] * 13)

    def test_trainable_false_excluded(self):
        rows = messages()
        rows[1]["trainable"] = False
        item = tokenize_with_mask(TinyTokenizer(), rows, [], 100)
        self.assertEqual(
            [x for x in item["labels"] if x != -100], [201, 2]
        )

    def test_template_end_token_supervised(self):
        item = tokenize_with_mask(
            TinyTokenizer(), [{"role": "assistant", "content": "x"}], [], 10
        )
        self.assertEqual(item["labels"], [-100, 220, 2])

    def test_clean_messages_preserves_input(self):
        rows = messages()
        rows[1]["trainable"] = True
        before = copy.deepcopy(rows)
        cleaned = clean_messages(rows)
        tokenize_with_mask(TinyTokenizer(), rows, [], 100)
        self.assertEqual(rows, before)
        self.assertNotIn("trainable", cleaned[1])

    def test_exact_max_length_accepted(self):
        item = tokenize_with_mask(TinyTokenizer(), messages(), [], 13)
        self.assertEqual(len(item["input_ids"]), 13)

    def test_overlength_rejected_without_truncation(self):
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            tokenize_with_mask(TinyTokenizer(), messages(), [], 12)

    def test_no_supervision_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "No supervised"):
            tokenize_with_mask(
                TinyTokenizer(), [{"role": "user", "content": "x"}], [], 100
            )

    def test_generation_prefix_change_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "generation prefix changed"):
            tokenize_with_mask(BadGenerationPrefix(), messages(), [], 100)

    def test_full_conversation_prefix_change_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Full conversation prefix"):
            tokenize_with_mask(BadConversationPrefix(), messages(), [], 100)


if __name__ == "__main__":
    unittest.main()
