import unittest
import json
from unittest.mock import MagicMock
from gutenberg.recipes_storage_and_retrieval_v2 import (
    generate_nutrition_info_chain,
    generate_shopping_list_chain,
    generate_factoids_chain
)

class TestRecipeIntegration(unittest.TestCase):

    def setUp(self):
        """Set up a mock LLM that returns predefined JSON responses."""
        self.mock_llm = MagicMock()

        # Ensuring all responses are valid JSON strings
        self.mock_llm.invoke.side_effect = lambda query: (
            json.dumps({
                "calories": 500,
                "protein": 30,
                "carbs": 50,
                "fat": 10
            }) if "nutrition" in query["text"] else 
            json.dumps({
                "items": ["rice", "chicken", "olive oil"]
            }) if "shopping" in query["text"] else 
            json.dumps({
                "factoids": [
                    "Rice has been cultivated for over 10,000 years.",
                    "Olive oil was called liquid gold by the ancient Greeks."
                ]
            }) if "factoids" in query["text"] else 
            json.dumps({})
        )  # Simulating different responses based on input

    def test_recipe_integration(self):
        """Test if all chains work together in a pipeline."""
        recipe_text = "nutrition: 1 cup rice, 100g chicken, 1 tbsp olive oil"

        # Generate nutrition info
        nutrition_chain = generate_nutrition_info_chain(self.mock_llm)
        nutrition_result = nutrition_chain.invoke({"text": recipe_text})
        
        self.assertIsInstance(nutrition_result, str, "Expected string JSON response")
        nutrition_data = json.loads(nutrition_result)
        self.assertTrue(all(key in nutrition_data for key in ["calories", "protein", "carbs", "fat"]),
                        "Missing expected keys in the nutrition JSON response")

        # Generate shopping list
        shopping_chain = generate_shopping_list_chain(self.mock_llm)
        shopping_result = shopping_chain.invoke({"text": "shopping: " + recipe_text})

        self.assertIsInstance(shopping_result, str, "Expected string JSON response")
        shopping_data = json.loads(shopping_result)
        self.assertIn("items", shopping_data)
        self.assertIsInstance(shopping_data["items"], list)

        # Generate factoids
        factoids_chain = generate_factoids_chain(self.mock_llm)
        factoids_result = factoids_chain.invoke({"text": "factoids: " + recipe_text})

        self.assertIsInstance(factoids_result, str, "Expected string JSON response")
        factoids_data = json.loads(factoids_result)
        self.assertIn("factoids", factoids_data)
        self.assertIsInstance(factoids_data["factoids"], list)

    def test_integration_with_empty_response(self):
        """Test the entire pipeline when the LLM returns empty responses."""
        self.mock_llm.invoke.return_value = json.dumps({})  # Ensuring an empty but valid JSON response

        recipe_text = "1 cup rice, 100g chicken, 1 tbsp olive oil"

        nutrition_chain = generate_nutrition_info_chain(self.mock_llm)
        shopping_chain = generate_shopping_list_chain(self.mock_llm)
        factoids_chain = generate_factoids_chain(self.mock_llm)

        self.assertEqual(nutrition_chain.invoke({"text": recipe_text}), "{}")
        self.assertEqual(shopping_chain.invoke({"text": recipe_text}), "{}")
        self.assertEqual(factoids_chain.invoke({"text": recipe_text}), "{}")

    def test_integration_with_malformed_json(self):
        """Test the entire pipeline when the LLM returns malformed JSON."""
        self.mock_llm.invoke.return_value = "{calories: 500, protein: 30"  # Malformed JSON response

        recipe_text = "1 cup rice, 100g chicken, 1 tbsp olive oil"

        nutrition_chain = generate_nutrition_info_chain(self.mock_llm)
        shopping_chain = generate_shopping_list_chain(self.mock_llm)
        factoids_chain = generate_factoids_chain(self.mock_llm)

        with self.assertRaises(json.JSONDecodeError):
            json.loads(nutrition_chain.invoke({"text": recipe_text}))

        with self.assertRaises(json.JSONDecodeError):
            json.loads(shopping_chain.invoke({"text": recipe_text}))

        with self.assertRaises(json.JSONDecodeError):
            json.loads(factoids_chain.invoke({"text": recipe_text}))

if __name__ == "__main__":
    unittest.main()


# Run: python -m unittest tests/test_augmentations_integration.py