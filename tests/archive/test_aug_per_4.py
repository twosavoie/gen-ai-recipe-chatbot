import unittest
import time
from unittest.mock import MagicMock
from gutenberg.recipes_storage_and_retrieval_v2 import generate_nutrition_info_chain

class PerformanceTestRecipeChatbotChains(unittest.TestCase):
    def setUp(self):
        self.ingredients_text = (
            "Ingredients:\n"
            "- 2 eggs\n"
            "- 1 cup of milk\n"
            "- 1 tbsp butter\n"
            "- 1/2 cup flour"
        )

    def test_nutrition_info_chain_performance(self):
        # Mock the LLM to return a valid JSON string
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"calories": 350, "protein": 20, "carbs": 25, "fat": 18}'
        )

        # Generate the chain with the mocked LLM
        nutrition_chain = generate_nutrition_info_chain(mock_llm)

        # Time the execution
        start_time = time.time()
        result = nutrition_chain.invoke({"text": self.ingredients_text})
        elapsed = time.time() - start_time

        # Ensure the result is parsed to a Python dict
        self.assertIsInstance(result, dict)
        self.assertIn("calories", result)
        self.assertIn("protein", result)
        self.assertIn("carbs", result)
        self.assertIn("fat", result)

        # Performance assertion (adjust as needed)
        self.assertLess(elapsed, 2.0, f"Performance test failed: took {elapsed:.2f}s")

if __name__ == "__main__":
    unittest.main()
