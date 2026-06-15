"""
tools.py

The three required FitFindr tools. Each tool is a standalone function that
can be called and tested independently before being wired into the agent loop.

Complete and test each tool before moving to agent.py.

Tools:
    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)              → str
    create_fit_card(outfit, new_item)               → str
"""

import os
from xmlrpc import client

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()


# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Args:
        description: Keywords describing what the user is looking for
                     (e.g., "vintage graphic tee").
        size:        Size string to filter by, or None to skip size filtering.
                     Matching is case-insensitive (e.g., "M" matches "S/M").
        max_price:   Maximum price (inclusive), or None to skip price filtering.

    Returns:
        A list of matching listing dicts, sorted by relevance (best match first).
        Returns an empty list if nothing matches — does NOT raise an exception.

    Each listing dict has the following fields:
        id, title, description, category, style_tags (list), size,
        condition, price (float), colors (list), brand, platform

    TODO:
        1. Load all listings with load_listings().
        2. Filter by max_price and size (if provided).
        3. Score each remaining listing by keyword overlap with `description`.
        4. Drop any listings with a score of 0 (no relevant matches).
        5. Sort by score, highest first, and return the listing dicts.

    Before writing code, fill in the Tool 1 section of planning.md.
    """
    listings = load_listings()
    results = []

    for item in listings:
        # 1. Filter by description (checks title, description, category, and tags)
        if description:
            # Combine relevant fields into one searchable string
            search_text = f"{item.get('title', '')} {item.get('description', '')} {item.get('category', '')} {' '.join(item.get('style_tags', []))}".lower()
            
            # --- UPDATED LOGIC: Keyword matching instead of exact phrase ---
            # Remove punctuation and split into words
            clean_desc = description.lower().replace('.', '').replace(',', '').replace('!', '')
            query_words = clean_desc.split()
            
            # Ignore common conversational filler words
            stop_words = {"find", "me", "some", "looking", "for", "i", "im", "i'm", "only", "wear", "and", "hate", "the", "a", "to", "want"}
            keywords = [w for w in query_words if w not in stop_words]
            
            # Check if ANY of our meaningful keywords are in the item's text
            if keywords:
                has_match = any(kw in search_text for kw in keywords)
                if not has_match:
                    continue
        
        # 2. Filter by size (exact match)
        if size and item.get('size') != size:
            continue
            
        # 3. Filter by maximum price
        if max_price and item.get('price', float('inf')) > max_price:
            continue
            
        # If it passes all criteria, add to results
        results.append(item)

    return results


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def suggest_outfit(new_item: dict, wardrobe: dict, style_memory: list = None) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty — handle this gracefully.

    Returns:
        A non-empty string with outfit suggestions.
        If the wardrobe is empty, offer general styling advice for the item
        rather than raising an exception or returning an empty string.

    TODO:
        1. Check whether wardrobe['items'] is empty.
        2. If empty: call the LLM with a prompt for general styling ideas
           (what kinds of items pair well, what vibe it suits, etc.).
        3. If not empty: format the wardrobe items into a prompt and ask
           the LLM to suggest specific outfit combinations using the new item
           and named pieces from the wardrobe.
        4. Return the LLM's response as a string.

    Before writing code, fill in the Tool 2 section of planning.md.
    """
    # 1. Handle the failure mode: Empty wardrobe
    if not wardrobe or not wardrobe.get("items"):
        return f"I see your wardrobe is empty right now! Generally, this {new_item.get('title', 'item')} would look great paired with some classic basics like wide-leg jeans and neutral sneakers."

    # --- STRETCH FEATURE: Inject Memory ---
    memory_context = ""
    if style_memory:
        memory_context = f"\nKeep in mind my overall style preferences: {', '.join(style_memory)}\n"

    # 2. Construct the prompt
    prompt = f"""
    You are an expert fashion stylist. I just thrifted this item:
    {new_item.get('title')} - {new_item.get('description')}

    Here is my current wardrobe:
    {wardrobe.get('items')}
    {memory_context}

    Suggest one complete outfit combining the new item with pieces from my wardrobe. 
    Keep it stylish, natural, and brief (2-3 sentences max). Do not use robotic intro phrases like "Here is a suggestion."
    """

    # 3. Call the LLM
    try:
        groq_client = _get_groq_client()
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, I had a little trouble pulling an outfit together right now. (Error: {str(e)})"


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Args:
        outfit:   The outfit suggestion string from suggest_outfit().
        new_item: The listing dict for the thrifted item.

    Returns:
        A 2–4 sentence string usable as an Instagram/TikTok caption.
        If outfit is empty or missing, return a descriptive error message
        string — do NOT raise an exception.

    The caption should:
    - Feel casual and authentic (like a real OOTD post, not a product description)
    - Mention the item name, price, and platform naturally (once each)
    - Capture the outfit vibe in specific terms
    - Sound different each time for different inputs (use higher LLM temperature)

    TODO:
        1. Guard against an empty or whitespace-only outfit string.
        2. Build a prompt that gives the LLM the item details and the outfit,
           and asks for a caption matching the style guidelines above.
        3. Call the LLM and return the response.

    Before writing code, fill in the Tool 3 section of planning.md.
    """
    # 1. Handle the failure mode: Empty or invalid outfit string
    if not outfit or not str(outfit).strip():
        return "Oops, I didn't get an outfit to caption! Let's try searching for a different item."

    # 2. Construct the prompt
    prompt = f"""
    You are a trendy fashion enthusiast. I just put together this outfit featuring a thrifted item:
    New Item: {new_item.get('title')}
    Outfit Details: {outfit}

    Write a short, catchy, natural-sounding caption for an Instagram post showing off this look.
    It should be 1-2 sentences max, use 1 or 2 emojis, and sound casual (no hashtags).
    Make it sound like a real person sharing their thrift find, mentioning the item.
    """

    # 3. Call the LLM
    try:
        groq_client = _get_groq_client()
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, I couldn't generate a fit card right now. (Error: {str(e)})"
    
def compare_price(item: dict) -> str:
    """
    Compares the item's price against the average price of its category.
    """
    from utils.data_loader import load_listings
    listings = load_listings()
    category = item.get("category")
    
    # Find all other items in the same category
    comparables = [l for l in listings if l.get("category") == category and l.get("id") != item.get("id")]
    
    if not comparables:
        return "Unique item! Hard to compare prices."
        
    avg_price = sum(l.get("price", 0) for l in comparables) / len(comparables)
    item_price = item.get("price", 0)
    
    if item_price < avg_price:
        return f"📉 Great deal! At ${item_price:.2f}, it is below the category average of ${avg_price:.2f}."
    else:
        return f"⚖️ Fair price. At ${item_price:.2f}, it is around the category average of ${avg_price:.2f}."

# Tool 5: extract_style_preferences (Stretch Feature)

def extract_style_preferences(query: str) -> str:
    """
    Extracts explicit style preferences from the user's query using the LLM.
    """
    prompt = f"""
    Extract any explicit style, color, or fit preferences from this query: "{query}"
    Examples of preferences: "I love vintage", "baggy fits only", "I hate bright colors".
    If there are no explicit style preferences, return exactly the word "None".
    Otherwise, return a short phrase summarizing the preference.
    """
    try:
        groq_client = _get_groq_client()
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "None"