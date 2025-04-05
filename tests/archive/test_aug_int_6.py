import unittest
from unittest.mock import MagicMock
import json
from langchain.schema import Document
from langchain.schema.messages import HumanMessage
from gutenberg.recipes_storage_and_retrieval_v2 import generate_nutrition_info_chain

class TestNutritionInfoChain(unittest.TestCase):
    def test_generate_nutrition_info_chain(self):
        # Create a proper mock LLM that behaves more like a real LLM
        nutrition_data = {
            "calories": 500,
            "protein": 30,
            "carbs": 60,
            "fat": 20
        }
        
        class MockLLM:
            def invoke(self, messages):
                # A real LLM would take messages and return content
                return json.dumps(nutrition_data)
                
        mock_llm = MockLLM()
        
        # Create a mock document with ingredients
        ingredients_text = "1 cup rice, 100g chicken, 1 tbsp olive oil"
        
        # Generate Nutrition Info Chain
        nutrition_chain = generate_nutrition_info_chain(mock_llm)
        
        #! remove this
        print(f"Type of mock_llm: {type(mock_llm)}")
        print(f"Type of ingredients_text: {type(ingredients_text)}")
        # Invoke the chain with the ingredients text
        result = nutrition_chain.invoke(ingredients_text)
        # result = nutrition_chain.invoke({"text": ingredients_text}) 

        
        # Parse the result
        parsed_data = json.loads(result)
        
        # Assert the parsed data
        self.assertEqual(parsed_data["calories"], 500)
        self.assertEqual(parsed_data["protein"], 30)
        self.assertEqual(parsed_data["carbs"], 60)
        self.assertEqual(parsed_data["fat"], 20)

if __name__ == "__main__":
    unittest.main()