import unittest
import json
from unittest.mock import MagicMock
from gutenberg.recipes_storage_and_retrieval_v2 import (
    generate_nutrition_info_chain,
    generate_shopping_list_chain,
    generate_factoids_chain
)

class TestRecipesStorageAndRetrieval(unittest.TestCase):

    ### ✅ Standard Tests ✅ ###

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

    ### ⚠️ Error Handling Tests for All Functions ⚠️ ###

    def test_generate_nutrition_info_chain_with_empty_string_response(self):
        mock_llm = MagicMock()
        mock_llm.return_value = ""

        chain = generate_nutrition_info_chain(mock_llm)
        result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

        self.assertEqual(result, "", "Expected an empty string when LLM response is empty.")

    def test_generate_shopping_list_chain_with_empty_string_response(self):
        mock_llm = MagicMock()
        mock_llm.return_value = ""

        chain = generate_shopping_list_chain(mock_llm)
        result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

        self.assertEqual(result, "", "Expected an empty string when LLM response is empty.")

    def test_generate_factoids_chain_with_empty_string_response(self):
        mock_llm = MagicMock()
        mock_llm.return_value = ""

        chain = generate_factoids_chain(mock_llm)
        result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

        self.assertEqual(result, "", "Expected an empty string when LLM response is empty.")

    def test_generate_nutrition_info_chain_with_malformed_json(self):
        mock_llm = MagicMock()
        mock_llm.return_value = "{calories: 500, protein: 30, carbs: 50, fat: 10"  # Missing closing bracket

        chain = generate_nutrition_info_chain(mock_llm)
        try:
            result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})
            json.loads(result)  # Should raise an error
            self.fail("Expected a JSONDecodeError due to malformed JSON.")
        except json.JSONDecodeError:
            pass

    def test_generate_shopping_list_chain_with_malformed_json(self):
        mock_llm = MagicMock()
        mock_llm.return_value = "{items: [rice, chicken, olive oil}"  # Missing closing bracket

        chain = generate_shopping_list_chain(mock_llm)
        try:
            result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})
            json.loads(result)
            self.fail("Expected a JSONDecodeError due to malformed JSON.")
        except json.JSONDecodeError:
            pass

    def test_generate_factoids_chain_with_malformed_json(self):
        mock_llm = MagicMock()
        mock_llm.return_value = "{factoids: [Rice has been cultivated for over 10,000 years.}"  # Missing closing bracket

        chain = generate_factoids_chain(mock_llm)
        try:
            result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})
            json.loads(result)
            self.fail("Expected a JSONDecodeError due to malformed JSON.")
        except json.JSONDecodeError:
            pass

    def test_generate_nutrition_info_chain_with_empty_json(self):
        mock_llm = MagicMock()
        mock_llm.return_value = "{}"

        chain = generate_nutrition_info_chain(mock_llm)
        result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

        self.assertEqual(result, "{}", "Expected an empty JSON object when LLM returns '{}'.")

    def test_generate_shopping_list_chain_with_empty_json(self):
        mock_llm = MagicMock()
        mock_llm.return_value = "{}"

        chain = generate_shopping_list_chain(mock_llm)
        result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

        self.assertEqual(result, "{}", "Expected an empty JSON object when LLM returns '{}'.")

    def test_generate_factoids_chain_with_empty_json(self):
        mock_llm = MagicMock()
        mock_llm.return_value = "{}"

        chain = generate_factoids_chain(mock_llm)
        result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

        self.assertEqual(result, "{}", "Expected an empty JSON object when LLM returns '{}'.")

if __name__ == "__main__":
    unittest.main()


# Run: python -m unittest tests.xtest_recipes_storage_and_retrieval_v2

# Which to use may depend on whether using StrOutpusParson or JsonOutputParser. I think the below failed because of a different output parser. Regardless, one should work. :) 

# import unittest
# import json
# from unittest.mock import MagicMock
# from gutenberg.recipes_storage_and_retrieval_v2 import (
#     generate_nutrition_info_chain,
#     generate_shopping_list_chain,
#     generate_factoids_chain
# )

# class TestRecipesStorageAndRetrieval(unittest.TestCase):

#     def test_generate_nutrition_info_chain(self):
#         mock_llm = MagicMock()
#         mock_llm.return_value = '{"calories": 500, "protein": 30, "carbs": 50, "fat": 10}'

#         chain = generate_nutrition_info_chain(mock_llm)
#         result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

#         mock_llm.assert_called()
#         self.assertEqual(result, '{"calories": 500, "protein": 30, "carbs": 50, "fat": 10}')

#     def test_generate_shopping_list_chain(self):
#         mock_llm = MagicMock()
#         mock_llm.return_value = '{"items": ["rice", "chicken", "olive oil"]}'

#         chain = generate_shopping_list_chain(mock_llm)
#         result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

#         mock_llm.assert_called()
#         self.assertEqual(result, '{"items": ["rice", "chicken", "olive oil"]}')

#     def test_generate_factoids_chain(self):
#         mock_llm = MagicMock()
#         mock_llm.return_value = '{"factoids": ["Rice has been cultivated for over 10,000 years.", "Olive oil was called liquid gold by the ancient Greeks."]}'

#         chain = generate_factoids_chain(mock_llm)
#         result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

#         mock_llm.assert_called()
#         self.assertEqual(
#             result,
#             '{"factoids": ["Rice has been cultivated for over 10,000 years.", "Olive oil was called liquid gold by the ancient Greeks."]}'
#         )

#     ### ERROR HANDLING TESTS ###

#     def test_generate_factoids_chain_with_none_response(self):
#         """Test when LLM returns None instead of a JSON response."""
#         mock_llm = MagicMock()
#         mock_llm.return_value = None  # Simulate failure

#         chain = generate_factoids_chain(mock_llm)
#         result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

#         self.assertIsNone(result, "Expected None when LLM response is None.")

#     def test_generate_factoids_chain_with_malformed_json(self):
#         """Test when LLM returns malformed JSON."""
#         mock_llm = MagicMock()
#         mock_llm.return_value = "{factoids: [Rice has been cultivated for over 10,000 years.}"  # Missing closing bracket

#         chain = generate_factoids_chain(mock_llm)
#         try:
#             result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})
#             json.loads(result)  # This should raise an error
#             self.fail("Expected a JSONDecodeError due to malformed JSON.")
#         except json.JSONDecodeError:
#             pass  # Expected behavior

#     def test_generate_factoids_chain_with_empty_json(self):
#         """Test when LLM returns an empty JSON object."""
#         mock_llm = MagicMock()
#         mock_llm.return_value = "{}"  # Empty response

#         chain = generate_factoids_chain(mock_llm)
#         result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

#         self.assertEqual(result, "{}", "Expected an empty JSON object if LLM returns empty JSON.")

# if __name__ == "__main__":
#     unittest.main()
# Run: python -m unittest discover tests