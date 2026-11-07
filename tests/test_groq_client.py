import os
import unittest
from unittest.mock import Mock, patch

from src.generation.groq_client import GroqClient


class GroqClientTests(unittest.TestCase):
    def test_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                GroqClient()

    @patch("src.generation.groq_client.requests.post")
    def test_adapts_chat_completion_response(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "Grounded answer"}}]
        }
        post.return_value = response
        client = GroqClient(api_key="test-key", timeout=12)
        result = client.chat(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": "Question"}],
            options={"temperature": 0},
            format="json",
        )
        self.assertEqual(result.message.content, "Grounded answer")
        request = post.call_args.kwargs
        self.assertEqual(request["timeout"], 12)
        self.assertEqual(request["json"]["response_format"], {"type": "json_object"})
        self.assertGreater(request["json"]["temperature"], 0)
        self.assertEqual(request["headers"]["Authorization"], "Bearer test-key")

    @patch("src.generation.groq_client.requests.post")
    def test_attaches_figure_image_to_prompt(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "Visual answer"}}]}
        post.return_value = response
        client = GroqClient(api_key="test-key", images=["data:image/png;base64,AAAA"])
        client.chat(model="qwen/qwen3.8-27b", messages=[{"role": "user", "content": "Explain Figure 2"}])
        content = post.call_args.kwargs["json"]["messages"][-1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Explain Figure 2"})
        self.assertEqual(content[1]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()
