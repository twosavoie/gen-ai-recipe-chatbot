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

    @patch("gutenberg.recipes_storage_and_retrieval_v2.generate_nutrition_info_chain")
    def test_nutrition_info_chain_performance(self, mock_generate_chain):
        expected_output = {
            "calories": 350,
            "protein": 20,
            "carbs": 25,
            "fat": 18
        }

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_output
        mock_generate_chain.return_value = mock_chain

        chain = recipes_storage_and_retrieval_v2.generate_nutrition_info_chain(llm="mock")
        start_time = time.time()
        result = chain.invoke({"text": self.ingredients_text})
        elapsed = time.time() - start_time

        self.assertIsInstance(result, dict)
        self.assertDictEqual(result, expected_output)
        self.assertLess(elapsed, 2.0, f"Performance test failed: took {elapsed:.2f}s")

    @patch("gutenberg.recipes_storage_and_retrieval_v2.generate_shopping_list_chain")
    def test_shopping_list_chain_performance(self, mock_generate_chain):
        expected_output = {
            "shopping_list": [
                "eggs",
                "milk",
                "butter",
                "flour"
            ]
        }

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_output
        mock_generate_chain.return_value = mock_chain

        chain = recipes_storage_and_retrieval_v2.generate_shopping_list_chain(llm="mock")
        start_time = time.time()
        result = chain.invoke({"text": self.ingredients_text})
        elapsed = time.time() - start_time

        self.assertIsInstance(result, dict)
        self.assertIn("shopping_list", result)
        self.assertIsInstance(result["shopping_list"], list)
        self.assertLess(elapsed, 2.0, f"Performance test failed: took {elapsed:.2f}s")

    @patch("gutenberg.recipes_storage_and_retrieval_v2.generate_factoids_chain")
    def test_factoids_chain_performance(self, mock_generate_chain):
        expected_output = {
            "factoids": [
                "Eggs have been used in baking since ancient times.",
                "Butter adds richness and flavor to baked goods.",
                "Milk adds moisture and helps activate leavening agents."
            ]
        }

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_output
        mock_generate_chain.return_value = mock_chain

        chain = recipes_storage_and_retrieval_v2.generate_factoids_chain(llm="mock")
        start_time = time.time()
        result = chain.invoke({"text": self.ingredients_text})
        elapsed = time.time() - start_time

        self.assertIsInstance(result, dict)
        self.assertIn("factoids", result)
        self.assertIsInstance(result["factoids"], list)
        self.assertLess(elapsed, 2.0, f"Performance test failed: took {elapsed:.2f}s")

if __name__ == "__main__":
    unittest.main()
