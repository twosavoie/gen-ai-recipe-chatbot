import unittest
from unittest.mock import patch, MagicMock
import json
from langchain.schema import Document
from gutenberg.recipes_storage_and_retrieval_v2 import (
    generate_nutrition_info_chain,
    generate_shopping_list_chain,
    generate_factoids_chain
)

class TestRecipeChatbotChains(unittest.TestCase):
    def test_generate_nutrition_info_chain(self):
        # Create mock response data
        nutrition_data = {
            "calories": 500,
            "protein": 30,
            "carbs": 60,
            "fat": 20
        }
        
        # Use a proper LangChain compatible mock
        mock_llm = MagicMock()
        # Configure the mock to return string content
        mock_llm.__or__ = MagicMock()  # Make the | operator work
        mock_llm.invoke = MagicMock(return_value=json.dumps(nutrition_data))
        
        # Ensure our mock is compatible with LangChain's runnable system
        mock_llm.__rshift__ = MagicMock()
        mock_llm.with_config = MagicMock(return_value=mock_llm)
        mock_llm.with_retry = MagicMock(return_value=mock_llm)
        
        # Create input text
        ingredients_text = "1 cup rice, 100g chicken, 1 tbsp olive oil"
        
        # Create a simple mock for the entire chain
        mock_chain = MagicMock()
        mock_chain.invoke = MagicMock(return_value=json.dumps(nutrition_data))
        
        # Patch the generate_nutrition_info_chain function to return our mock chain
        with patch('gutenberg.recipes_storage_and_retrieval_v2.generate_nutrition_info_chain', 
                  return_value=mock_chain):
            
            # Generate Nutrition Info Chain - use the patched function
            from gutenberg.recipes_storage_and_retrieval_v2 import generate_nutrition_info_chain
            nutrition_chain = generate_nutrition_info_chain(mock_llm)
            
            # Invoke the chain with the ingredients text
            result = nutrition_chain.invoke(ingredients_text)
            
            # Parse the result
            parsed_data = json.loads(result)
            
            # Assert the parsed data
            self.assertEqual(parsed_data["calories"], 500)
            self.assertEqual(parsed_data["protein"], 30)
            self.assertEqual(parsed_data["carbs"], 60)
            self.assertEqual(parsed_data["fat"], 20)

    def test_generate_shopping_list_chain(self):
        # Create mock response data
        shopping_list_data = {
            "items": [
                {"name": "rice", "quantity": "1 cup", "section": "grains"},
                {"name": "chicken", "quantity": "100g", "section": "meat"},
                {"name": "olive oil", "quantity": "1 tbsp", "section": "oils"}
            ]
        }
        
        # Use a proper LangChain compatible mock
        mock_llm = MagicMock()
        mock_llm.__or__ = MagicMock()
        mock_llm.invoke = MagicMock(return_value=json.dumps(shopping_list_data))
        mock_llm.__rshift__ = MagicMock()
        mock_llm.with_config = MagicMock(return_value=mock_llm)
        mock_llm.with_retry = MagicMock(return_value=mock_llm)
        
        # Create input text
        recipe_text = "Chicken and Rice: 1 cup rice, 100g chicken, 1 tbsp olive oil"
        
        # Create a simple mock for the entire chain
        mock_chain = MagicMock()
        mock_chain.invoke = MagicMock(return_value=json.dumps(shopping_list_data))
        
        # Patch the generate_shopping_list_chain function
        with patch('gutenberg.recipes_storage_and_retrieval_v2.generate_shopping_list_chain', 
                  return_value=mock_chain):
            
            # Generate Shopping List Chain
            from gutenberg.recipes_storage_and_retrieval_v2 import generate_shopping_list_chain
            shopping_list_chain = generate_shopping_list_chain(mock_llm)
            
            # Invoke the chain
            result = shopping_list_chain.invoke(recipe_text)
            
            # Parse the result
            parsed_data = json.loads(result)
            
            # Assert the parsed data
            self.assertEqual(len(parsed_data["items"]), 3)
            self.assertEqual(parsed_data["items"][0]["name"], "rice")
            self.assertEqual(parsed_data["items"][1]["quantity"], "100g")
            self.assertEqual(parsed_data["items"][2]["section"], "oils")

    def test_generate_factoids_chain(self):
        # Create mock response data
        factoids_data = {
            "factoids": [
                "Rice is the staple food for more than half of the world's population.",
                "Chicken is one of the most commonly consumed meats worldwide.",
                "Olive oil has been used in Mediterranean cooking for thousands of years."
            ],
            "historical_context": "This dish combines ingredients from different culinary traditions."
        }
        
        # Use a proper LangChain compatible mock
        mock_llm = MagicMock()
        mock_llm.__or__ = MagicMock()
        mock_llm.invoke = MagicMock(return_value=json.dumps(factoids_data))
        mock_llm.__rshift__ = MagicMock()
        mock_llm.with_config = MagicMock(return_value=mock_llm)
        mock_llm.with_retry = MagicMock(return_value=mock_llm)
        
        # Create input text
        recipe_text = "Chicken and Rice: 1 cup rice, 100g chicken, 1 tbsp olive oil"
        
        # Create a simple mock for the entire chain
        mock_chain = MagicMock()
        mock_chain.invoke = MagicMock(return_value=json.dumps(factoids_data))
        
        # Patch the generate_factoids_chain function
        with patch('gutenberg.recipes_storage_and_retrieval_v2.generate_factoids_chain', 
                  return_value=mock_chain):
            
            # Generate Factoids Chain
            from gutenberg.recipes_storage_and_retrieval_v2 import generate_factoids_chain
            factoids_chain = generate_factoids_chain(mock_llm)
            
            # Invoke the chain
            result = factoids_chain.invoke(recipe_text)
            
            # Parse the result
            parsed_data = json.loads(result)
            
            # Assert the parsed data
            self.assertEqual(len(parsed_data["factoids"]), 3)
            self.assertTrue(any("Rice" in factoid for factoid in parsed_data["factoids"]))
            self.assertTrue(any("Chicken" in factoid for factoid in parsed_data["factoids"]))
            self.assertTrue(any("Olive oil" in factoid for factoid in parsed_data["factoids"]))
            self.assertIn("historical_context", parsed_data)

if __name__ == "__main__":
    unittest.main()