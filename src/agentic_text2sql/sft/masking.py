"""Assistant-only supervision extracted from the historical SFT trainer."""
from __future__ import annotations


def clean_messages(messages: list[dict]) -> list[dict]:
    return [
        {
            key: value
            for key, value in message.items()
            if key != "trainable"
        }
        for message in messages
    ]


def tokenize_with_mask(
    tokenizer,
    messages: list[dict],
    tools: list[dict],
    max_length: int,
) -> dict[str, list[int]]:
    cleaned = clean_messages(messages)

    input_ids = list(
        tokenizer.apply_chat_template(
            cleaned,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
        )
    )

    labels = [-100] * len(input_ids)

    for index, message in enumerate(messages):
        trainable = (
            message["role"] == "assistant"
            and message.get("trainable", True)
        )

        if not trainable:
            continue

        before_ids = list(
            tokenizer.apply_chat_template(
                clean_messages(
                    messages[:index]
                ),
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
            )
        )

        through_ids = list(
            tokenizer.apply_chat_template(
                clean_messages(
                    messages[: index + 1]
                ),
                tools=tools,
                tokenize=True,
                add_generation_prompt=False,
            )
        )

        if (
            through_ids[: len(before_ids)]
            != before_ids
        ):
            raise RuntimeError(
                "Assistant generation prefix changed "
                f"at message {index}"
            )

        if (
            input_ids[: len(through_ids)]
            != through_ids
        ):
            raise RuntimeError(
                "Full conversation prefix changed "
                f"at message {index}"
            )

        start = len(before_ids)
        end = len(through_ids)

        labels[start:end] = input_ids[start:end]

    if len(input_ids) > max_length:
        raise RuntimeError(
            f"Sequence length {len(input_ids)} "
            f"exceeds {max_length}"
        )

    if not any(
        label != -100
        for label in labels
    ):
        raise RuntimeError(
            "No supervised tokens"
        )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }
