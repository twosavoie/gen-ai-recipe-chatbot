import unittest
from unittest.mock import MagicMock
from gutenberg.recipes_storage_and_retrieval_v2 import (
    generate_nutrition_info_chain,
    generate_shopping_list_chain,
    generate_factoids_chain
)

class TestRecipesStorageAndRetrieval(unittest.TestCase):

    def test_generate_nutrition_info_chain(self):
        mock_llm = MagicMock()
        mock_llm.return_value = '{"calories": 500, "protein": 30, "carbs": 50, "fat": 10}'

        chain = generate_nutrition_info_chain(mock_llm)
        result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

        mock_llm.assert_called()
        self.assertEqual(result, '{"calories": 500, "protein": 30, "carbs": 50, "fat": 10}')

    def test_generate_shopping_list_chain(self):
        mock_llm = MagicMock()
        mock_llm.return_value = '{"items": ["rice", "chicken", "olive oil"]}'

        chain = generate_shopping_list_chain(mock_llm)
        result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

        mock_llm.assert_called()
        self.assertEqual(result, '{"items": ["rice", "chicken", "olive oil"]}')

    def test_generate_factoids_chain(self):
        mock_llm = MagicMock()
        mock_llm.return_value = '{"factoids": ["Rice has been cultivated for over 10,000 years.", "Olive oil was called liquid gold by the ancient Greeks."]}'

        chain = generate_factoids_chain(mock_llm)
        result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

        mock_llm.assert_called()
        self.assertEqual(
            result,
            '{"factoids": ["Rice has been cultivated for over 10,000 years.", "Olive oil was called liquid gold by the ancient Greeks."]}'
        )

if __name__ == "__main__":
    unittest.main()
    
# Run: python -m unittest discover tests
# Run: python -m unittest tests.test_recipes_storage_and_retrieval_v2

