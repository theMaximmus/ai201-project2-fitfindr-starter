# tests/test_tools.py
from tools import search_listings

from tools import suggest_outfit
from utils.data_loader import get_empty_wardrobe, get_example_wardrobe

from tools import create_fit_card

def test_search_returns_results():
    results = search_listings("vintage graphic tee", size=None, max_price=50)
    assert isinstance(results, list)
    assert len(results) > 0

def test_search_empty_results():
    results = search_listings("designer ballgown", size="XXS", max_price=5)
    assert results == []  # empty list, no exception

def test_search_price_filter():
    results = search_listings("jacket", size=None, max_price=10)
    assert all(item["price"] <= 10 for item in results)

def test_suggest_outfit_empty_wardrobe():
    mock_item = {"title": "Vintage Band Tee", "description": "Faded black graphic tee"}
    empty_wardrobe = get_empty_wardrobe()
    
    result = suggest_outfit(mock_item, empty_wardrobe)
    
    assert isinstance(result, str)
    assert "empty" in result.lower() # Checks that our graceful error message triggered

def test_suggest_outfit_with_items():
    mock_item = {"title": "Vintage Band Tee", "description": "Faded black graphic tee"}
    example_wardrobe = get_example_wardrobe()
    
    result = suggest_outfit(mock_item, example_wardrobe)
    
    assert isinstance(result, str)
    assert len(result) > 20 # Ensures the LLM actually returned a substantial string

def test_fit_card_empty_outfit():
    mock_item = {"title": "Vintage Band Tee"}
    
    # Passing an empty string instead of a real outfit
    result = create_fit_card("", mock_item) 
    
    assert isinstance(result, str)
    assert "Oops" in result

def test_fit_card_valid_outfit():
    mock_item = {"title": "Vintage Band Tee"}
    mock_outfit = "Pair this tee with baggy wide-leg jeans and platform docs."
    
    result = create_fit_card(mock_outfit, mock_item)
    
    assert isinstance(result, str)
    assert len(result) > 15