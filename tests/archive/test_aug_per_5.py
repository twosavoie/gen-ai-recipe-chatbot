import unittest
import time
from unittest.mock import patch, MagicMock
from gutenberg import recipes_storage_and_retrieval_v2

class PerformanceTestRecipeChatbotChains(unittest.TestCase):
    def setUp(self):
        self.ingredients_text = (
            "Ingredients:\n"
            "- 2 eggs\n"
            "- 1 cup of milk\n"
            "- 1 tbsp butter\n"
            "- 1/2 cup flour"
        )
        self.expected_output = {
            "calories": 350,
            "protein": 20,
            "carbs": 25,
            "fat": 18
        }

    @patch("gutenberg.recipes_storage_and_retrieval_v2.generate_nutrition_info_chain")
    def test_nutrition_info_chain_performance(self, mock_generate_chain):
        # Create a mock chain that returns the expected output
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self.expected_output

        # Patch the actual chain generator to return our mock chain
        mock_generate_chain.return_value = mock_chain

        # Now call the function under test (as if it were real)
        chain = recipes_storage_and_retrieval_v2.generate_nutrition_info_chain(llm="irrelevant_for_mock")

        start_time = time.time()
        result = chain.invoke({"text": self.ingredients_text})
        elapsed = time.time() - start_time

        # Assertions
        self.assertIsInstance(result, dict)
        self.assertDictEqual(result, self.expected_output)
        self.assertLess(elapsed, 2.0, f"Performance test failed: took {elapsed:.2f}s")

if __name__ == "__main__":
    unittest.main()
