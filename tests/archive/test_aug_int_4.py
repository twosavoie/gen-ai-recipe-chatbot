import unittest
from unittest.mock import MagicMock
import json
from langchain.schema import Document
from gutenberg.recipes_storage_and_retrieval_v2 import generate_nutrition_info_chain

class TestNutritionInfoChain(unittest.TestCase):
    def test_generate_nutrition_info_chain(self):
        # Mock LLM with proper return format
        mock_llm = MagicMock()
        mock_response = json.dumps({
            "calories": 500,
            "protein": 30,
            "carbs": 60,
            "fat": 20
        })
        # Configure the mock to return a string directly
        mock_llm.invoke.return_value = mock_response
        
        # Create a mock document with ingredients
        ingredients_text = "1 cup rice, 100g chicken, 1 tbsp olive oil"
        document = Document(page_content=ingredients_text, metadata={})
        
        # Generate Nutrition Info Chain
        nutrition_chain = generate_nutrition_info_chain(mock_llm)
        
        # Modify this part to match the expected input/output format of your chain
        # Option 1: If your chain expects a Document but returns a string
        try:
            result = nutrition_chain.invoke(document)
            # Parse the result
            nutrition_data = json.loads(result)
        except TypeError:
            # Option 2: If your chain expects the text content of the document
            result = nutrition_chain.invoke(document.page_content)
            nutrition_data = json.loads(result)
        
        # Assert the parsed data
        self.assertEqual(nutrition_data["calories"], 500)
        self.assertEqual(nutrition_data["protein"], 30)
        self.assertEqual(nutrition_data["carbs"], 60)
        self.assertEqual(nutrition_data["fat"], 20)

if __name__ == "__main__":
    unittest.main()