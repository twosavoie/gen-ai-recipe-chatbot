import unittest
import time
# import psutil
import json
import statistics
from unittest.mock import patch, MagicMock
import tracemalloc
from gutenberg.recipes_storage_and_retrieval_v2 import (
    generate_nutrition_info_chain,
    generate_shopping_list_chain,
    generate_factoids_chain
)

class PerformanceTestRecipeChatbotChains(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.mock_llm = MagicMock()
        self.mock_llm.__or__ = MagicMock()
        self.mock_llm.__rshift__ = MagicMock()
        self.mock_llm.with_config = MagicMock(return_value=self.mock_llm)
        self.mock_llm.with_retry = MagicMock(return_value=self.mock_llm)
        
        # Sample input text
        self.ingredients_text = "1 cup rice, 100g chicken, 1 tbsp olive oil, 2 cloves garlic, 1/2 onion, 1 cup chicken broth"
        
        # Performance thresholds
        self.max_execution_time = 0.5  # seconds
        self.max_memory_increase = 5 * 1024 * 1024  # 5 MB in bytes
        
        # Number of iterations for performance testing
        self.iterations = 10
    
    def test_nutrition_info_chain_performance(self):
        """Test performance of the nutrition info chain."""
        # Mock data
        nutrition_data = {
            "calories": 500,
            "protein": 30,
            "carbs": 60,
            "fat": 20
        }
        
        # Configure mock
        self.mock_llm.invoke = MagicMock(return_value=json.dumps(nutrition_data))
        mock_chain = MagicMock()
        mock_chain.invoke = MagicMock(return_value=json.dumps(nutrition_data))
        
        with patch('gutenberg.recipes_storage_and_retrieval_v2.generate_nutrition_info_chain', 
                  return_value=mock_chain):
            
            # Generate Chain
            nutrition_chain = generate_nutrition_info_chain(self.mock_llm)
            
            # Perform warm-up call
            _ = nutrition_chain.invoke(self.ingredients_text)
            
            # Measure execution time over multiple iterations
            execution_times = []
            
            # Start memory tracking
            tracemalloc.start()
            start_snapshot = tracemalloc.take_snapshot()
            
            for _ in range(self.iterations):
                start_time = time.time()
                _ = nutrition_chain.invoke(self.ingredients_text)
                end_time = time.time()
                execution_times.append(end_time - start_time)
            
            # Measure memory usage
            end_snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            
            # Calculate memory difference
            memory_diff = sum(stat.size for stat in end_snapshot.compare_to(start_snapshot, 'lineno'))
            
            # Calculate statistics
            avg_time = statistics.mean(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            
            # Print performance metrics
            print(f"\nNutrition Info Chain Performance:")
            print(f"Average execution time: {avg_time:.6f} seconds")
            print(f"Min execution time: {min_time:.6f} seconds")
            print(f"Max execution time: {max_time:.6f} seconds")
            print(f"Memory increase: {memory_diff / (1024 * 1024):.2f} MB")
            
            # Assert performance meets expectations
            self.assertLess(avg_time, self.max_execution_time, 
                           f"Average execution time {avg_time:.6f}s exceeds threshold {self.max_execution_time}s")
            self.assertLess(memory_diff, self.max_memory_increase,
                           f"Memory increase {memory_diff/(1024*1024):.2f}MB exceeds threshold {self.max_memory_increase/(1024*1024)}MB")
    
    def test_shopping_list_chain_performance(self):
        """Test performance of the shopping list chain."""
        # Mock data
        shopping_list_data = {
            "items": [
                {"name": "rice", "quantity": "1 cup", "section": "grains"},
                {"name": "chicken", "quantity": "100g", "section": "meat"},
                {"name": "olive oil", "quantity": "1 tbsp", "section": "oils"},
                {"name": "garlic", "quantity": "2 cloves", "section": "produce"},
                {"name": "onion", "quantity": "1/2", "section": "produce"},
                {"name": "chicken broth", "quantity": "1 cup", "section": "canned goods"}
            ]
        }
        
        # Configure mock
        self.mock_llm.invoke = MagicMock(return_value=json.dumps(shopping_list_data))
        mock_chain = MagicMock()
        mock_chain.invoke = MagicMock(return_value=json.dumps(shopping_list_data))
        
        with patch('gutenberg.recipes_storage_and_retrieval_v2.generate_shopping_list_chain', 
                  return_value=mock_chain):
            
            # Generate Chain
            shopping_list_chain = generate_shopping_list_chain(self.mock_llm)
            
            # Perform warm-up call
            _ = shopping_list_chain.invoke(self.ingredients_text)
            
            # Measure execution time over multiple iterations
            execution_times = []
            
            # Start memory tracking
            tracemalloc.start()
            start_snapshot = tracemalloc.take_snapshot()
            
            for _ in range(self.iterations):
                start_time = time.time()
                _ = shopping_list_chain.invoke(self.ingredients_text)
                end_time = time.time()
                execution_times.append(end_time - start_time)
            
            # Measure memory usage
            end_snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            
            # Calculate memory difference
            memory_diff = sum(stat.size for stat in end_snapshot.compare_to(start_snapshot, 'lineno'))
            
            # Calculate statistics
            avg_time = statistics.mean(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            
            # Print performance metrics
            print(f"\nShopping List Chain Performance:")
            print(f"Average execution time: {avg_time:.6f} seconds")
            print(f"Min execution time: {min_time:.6f} seconds")
            print(f"Max execution time: {max_time:.6f} seconds")
            print(f"Memory increase: {memory_diff / (1024 * 1024):.2f} MB")
            
            # Assert performance meets expectations
            self.assertLess(avg_time, self.max_execution_time, 
                           f"Average execution time {avg_time:.6f}s exceeds threshold {self.max_execution_time}s")
            self.assertLess(memory_diff, self.max_memory_increase,
                           f"Memory increase {memory_diff/(1024*1024):.2f}MB exceeds threshold {self.max_memory_increase/(1024*1024)}MB")
    
    def test_factoids_chain_performance(self):
        """Test performance of the factoids chain."""
        # Mock data
        factoids_data = {
            "factoids": [
                "Rice is the staple food for more than half of the world's population.",
                "Chicken is one of the most commonly consumed meats worldwide.",
                "Olive oil has been used in Mediterranean cooking for thousands of years.",
                "Garlic has been used for both culinary and medicinal purposes for centuries.",
                "Onions are one of the oldest cultivated vegetables in human history."
            ],
            "historical_context": "This dish combines ingredients from different culinary traditions."
        }
        
        # Configure mock
        self.mock_llm.invoke = MagicMock(return_value=json.dumps(factoids_data))
        mock_chain = MagicMock()
        mock_chain.invoke = MagicMock(return_value=json.dumps(factoids_data))
        
        with patch('gutenberg.recipes_storage_and_retrieval_v2.generate_factoids_chain', 
                  return_value=mock_chain):
            
            # Generate Chain
            factoids_chain = generate_factoids_chain(self.mock_llm)
            
            # Perform warm-up call
            _ = factoids_chain.invoke(self.ingredients_text)
            
            # Measure execution time over multiple iterations
            execution_times = []
            
            # Start memory tracking
            tracemalloc.start()
            start_snapshot = tracemalloc.take_snapshot()
            
            for _ in range(self.iterations):
                start_time = time.time()
                _ = factoids_chain.invoke(self.ingredients_text)
                end_time = time.time()
                execution_times.append(end_time - start_time)
            
            # Measure memory usage
            end_snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            
            # Calculate memory difference
            memory_diff = sum(stat.size for stat in end_snapshot.compare_to(start_snapshot, 'lineno'))
            
            # Calculate statistics
            avg_time = statistics.mean(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            
            # Print performance metrics
            print(f"\nFactoids Chain Performance:")
            print(f"Average execution time: {avg_time:.6f} seconds")
            print(f"Min execution time: {min_time:.6f} seconds")
            print(f"Max execution time: {max_time:.6f} seconds")
            print(f"Memory increase: {memory_diff / (1024 * 1024):.2f} MB")
            
            # Assert performance meets expectations
            self.assertLess(avg_time, self.max_execution_time, 
                           f"Average execution time {avg_time:.6f}s exceeds threshold {self.max_execution_time}s")
            self.assertLess(memory_diff, self.max_memory_increase,
                           f"Memory increase {memory_diff/(1024*1024):.2f}MB exceeds threshold {self.max_memory_increase/(1024*1024)}MB")
    
    def test_comparative_performance(self):
        """Compare performance between all three chains."""
        # Setup mock data
        nutrition_data = {"calories": 500, "protein": 30, "carbs": 60, "fat": 20}
        shopping_list_data = {
            "items": [
                {"name": "rice", "quantity": "1 cup", "section": "grains"},
                {"name": "chicken", "quantity": "100g", "section": "meat"},
                {"name": "olive oil", "quantity": "1 tbsp", "section": "oils"}
            ]
        }
        factoids_data = {
            "factoids": [
                "Rice is the staple food for more than half of the world's population.",
                "Chicken is one of the most commonly consumed meats worldwide.",
                "Olive oil has been used in Mediterranean cooking for thousands of years."
            ],
            "historical_context": "This dish combines ingredients from different culinary traditions."
        }
        
        # Setup chains with mocks
        mock_nutrition_chain = MagicMock()
        mock_nutrition_chain.invoke = MagicMock(return_value=json.dumps(nutrition_data))
        
        mock_shopping_chain = MagicMock()
        mock_shopping_chain.invoke = MagicMock(return_value=json.dumps(shopping_list_data))
        
        mock_factoids_chain = MagicMock()
        mock_factoids_chain.invoke = MagicMock(return_value=json.dumps(factoids_data))
        
        # Performance results storage
        performance_results = {}
        
        # Test each chain
        with patch('gutenberg.recipes_storage_and_retrieval_v2.generate_nutrition_info_chain', 
                  return_value=mock_nutrition_chain):
            
            nutrition_chain = generate_nutrition_info_chain(self.mock_llm)
            
            # Warm-up
            _ = nutrition_chain.invoke(self.ingredients_text)
            
            # Test performance
            start_time = time.time()
            for _ in range(self.iterations):
                _ = nutrition_chain.invoke(self.ingredients_text)
            end_time = time.time()
            
            performance_results['nutrition'] = (end_time - start_time) / self.iterations
        
        with patch('gutenberg.recipes_storage_and_retrieval_v2.generate_shopping_list_chain', 
                  return_value=mock_shopping_chain):
            
            shopping_list_chain = generate_shopping_list_chain(self.mock_llm)
            
            # Warm-up
            _ = shopping_list_chain.invoke(self.ingredients_text)
            
            # Test performance
            start_time = time.time()
            for _ in range(self.iterations):
                _ = shopping_list_chain.invoke(self.ingredients_text)
            end_time = time.time()
            
            performance_results['shopping'] = (end_time - start_time) / self.iterations
        
        with patch('gutenberg.recipes_storage_and_retrieval_v2.generate_factoids_chain', 
                  return_value=mock_factoids_chain):
            
            factoids_chain = generate_factoids_chain(self.mock_llm)
            
            # Warm-up
            _ = factoids_chain.invoke(self.ingredients_text)
            
            # Test performance
            start_time = time.time()
            for _ in range(self.iterations):
                _ = factoids_chain.invoke(self.ingredients_text)
            end_time = time.time()
            
            performance_results['factoids'] = (end_time - start_time) / self.iterations
        
        # Print comparative results
        print("\nComparative Performance Analysis:")
        for chain, avg_time in performance_results.items():
            print(f"{chain.capitalize()} chain average time: {avg_time:.6f} seconds")
        
        # Find fastest and slowest chains
        fastest_chain = min(performance_results.items(), key=lambda x: x[1])
        slowest_chain = max(performance_results.items(), key=lambda x: x[1])
        
        print(f"\nFastest chain: {fastest_chain[0].capitalize()} ({fastest_chain[1]:.6f}s)")
        print(f"Slowest chain: {slowest_chain[0].capitalize()} ({slowest_chain[1]:.6f}s)")
        print(f"Performance ratio (slowest/fastest): {slowest_chain[1]/fastest_chain[1]:.2f}x")

if __name__ == "__main__":
    unittest.main()