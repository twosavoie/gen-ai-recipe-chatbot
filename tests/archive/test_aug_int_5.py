import unittest
from unittest.mock import MagicMock
import json
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from gutenberg.recipes_storage_and_retrieval_v2 import generate_nutrition_info_chain

class TestNutritionInfoChain(unittest.TestCase):
    def test_generate_nutrition_info_chain(self):
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({
            "calories": 500,
            "protein": 30,
            "carbs": 60,
            "fat": 20
        })
        
        # Create a mock document with ingredients
        ingredients_text = "1 cup rice, 100g chicken, 1 tbsp olive oil"
        document = Document(page_content=ingredients_text, metadata={})
        
        # Generate Nutrition Info Chain
        nutrition_chain = generate_nutrition_info_chain(mock_llm)
        
        # The chain expects text input, not a Document object
        result = nutrition_chain.invoke(ingredients_text)
        
        # Parse the result
        nutrition_data = json.loads(result)
        
        # Assert the parsed data
        self.assertEqual(nutrition_data["calories"], 500)
        self.assertEqual(nutrition_data["protein"], 30)
        self.assertEqual(nutrition_data["carbs"], 60)
        self.assertEqual(nutrition_data["fat"], 20)

if __name__ == "__main__":
    unittest.main()