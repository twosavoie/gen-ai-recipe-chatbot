
import unittest
import json
from unittest.mock import MagicMock
from gutenberg.recipes_storage_and_retrieval_v2 import (
    generate_nutrition_info_chain,
    generate_shopping_list_chain,
    generate_factoids_chain,
    build_outputs
)
# from langchain_core.schema import Document
from langchain.schema import Document

class TestRecipeIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()

    def test_recipe_integration(self):
        """Test if all chains work together in a pipeline."""
        recipe_text = "1 cup rice, 100g chicken, 1 tbsp olive oil"
        recipe_doc = Document(page_content=recipe_text, metadata={})
        
        # Mock LLM responses
        self.mock_llm.invoke = MagicMock(side_effect=[
            json.dumps({"calories": 300, "protein": 20, "carbs": 50, "fat": 10}),
            json.dumps({"items": ["rice", "chicken", "olive oil"]}),
            json.dumps({"factoids": ["Rice is a staple food in many cultures."]}),
            recipe_text
        ])
        
        results = [recipe_doc]
        outputs = build_outputs(results, self.mock_llm)

        self.assertEqual(len(outputs), 1)
        
        output = outputs[0]
        self.assertIn("nutrition", output)
        self.assertIn("shopping_list", output)
        self.assertIn("factoids", output)
        self.assertIn("recipe", output)

        nutrition_data = json.loads(output["nutrition"])
        self.assertEqual(nutrition_data["calories"], 300)
        self.assertEqual(nutrition_data["protein"], 20)
        self.assertEqual(nutrition_data["carbs"], 50)
        self.assertEqual(nutrition_data["fat"], 10)

        shopping_data = json.loads(output["shopping_list"])
        self.assertIn("rice", shopping_data["items"])
        self.assertIn("chicken", shopping_data["items"])
        self.assertIn("olive oil", shopping_data["items"])

        factoids_data = json.loads(output["factoids"])
        self.assertIn("Rice is a staple food in many cultures.", factoids_data["factoids"])

if __name__ == '__main__':
    unittest.main()
