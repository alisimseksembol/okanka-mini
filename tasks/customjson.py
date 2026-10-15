"""
CustomJSON task (Okanka Mini addition, modeled on the CustomJSON task that older nanochat
versions used for identity_conversations.jsonl).

Each line of the file is a JSON array of messages: [{"role": "user"|"assistant", "content": str}, ...]
(an optional leading "system" message is also accepted).
"""
import os
import json

from tasks.common import Task


class CustomJSON(Task):

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        assert os.path.exists(filepath), f"CustomJSON file not found: {filepath}"
        self.conversations = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                messages = json.loads(line)
                if isinstance(messages, dict):
                    messages = messages["messages"]
                rest = messages[1:] if messages[0]["role"] == "system" else messages
                assert len(rest) >= 2, f"conversation too short: {messages}"
                for i, m in enumerate(rest):
                    assert m["role"] == ("user" if i % 2 == 0 else "assistant"), f"bad role order: {messages}"
                    assert isinstance(m["content"], str)
                self.conversations.append(messages)
        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return {"messages": self.conversations[index]}
