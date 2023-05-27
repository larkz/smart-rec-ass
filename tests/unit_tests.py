import pytest
from unittest.mock import Mock
from modules.ml_tools import generate_embeddings, heuristic_rules
import spacy

def test_generate_embeddings():
    """
    Test the generate_embeddings function.

    This function verifies the correctness of the generate_embeddings function by checking if the number of embeddings matches the input data.

    """
    # Mock data
    data = [
        {"level": "Senior Level", "title": "Title 1", "description": "Description 1"},
        {"level": "Entry Level", "title": "Title 2", "description": "Description 2"},
        # Add more data entries as needed
    ]
    nlp = spacy.load("en_core_web_md")

    embeddings = generate_embeddings(data, nlp = nlp) # Mock data

    assert len(embeddings) == len(data)  # Check if the number of embeddings matches the input data
    # Add more assertions as needed to verify the correctness of the embeddings

def test_heuristic_rules():
    """
    Test the heuristic_rules function.

    This function verifies the correctness of the heuristic_rules function by checking if the inferred level matches the expected level for different input strings.

    """
    # Test cases
    test_cases = [
        ("Senior Software Engineer", 3),
        ("Head of Marketing", 3),
        ("Junior Analyst", 1),
        ("Internship Position", 0),
        # Add more test cases as needed
    ]

    for string_input, expected_level in test_cases:
        inferred_level = heuristic_rules(string_input)
        assert inferred_level == expected_level