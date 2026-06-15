"""
app.py

Gradio interface for FitFindr. The layout and wiring are already set up —
your job is to fill in handle_query() so it calls run_agent() and maps
the session results to the three output panels.

Run with:
    python app.py

Then open the localhost URL shown in your terminal (usually http://localhost:7860,
but check your terminal — the port may differ).
"""

import gradio as gr

from agent import run_agent
from utils.data_loader import get_example_wardrobe, get_empty_wardrobe


# ── query handler ─────────────────────────────────────────────────────────────

def handle_query(user_query: str, wardrobe_choice: str) -> tuple[str, str, str]:
    """
    Called by Gradio when the user submits a query.

    Args:
        user_query:     The text the user typed into the search box.
        wardrobe_choice: Either "Example wardrobe" or "Empty wardrobe (new user)".

    Returns:
        A tuple of three strings:
            (listing_text, outfit_suggestion, fit_card)
        Each string maps to one of the three output panels in the UI.

    TODO:
        1. Guard against an empty query (return early with an error message).
        2. Select the wardrobe based on wardrobe_choice.
        3. Call run_agent() with the query and selected wardrobe.
        4. If session["error"] is set, return the error in the first panel
           and empty strings for the other two.
        5. Otherwise, format session["selected_item"] into a readable listing_text
           string and return it along with session["outfit_suggestion"] and
           session["fit_card"].
    """
    # 1. Guard against an empty query
    if not user_query or not user_query.strip():
        return "Please enter a search description first!", "", ""

    # 2. Select the correct wardrobe based on the UI dropdown
    if wardrobe_choice == "Example wardrobe":
        wardrobe = get_example_wardrobe()
    else:
        wardrobe = get_empty_wardrobe()

    # 3. Call your agent!
    session = run_agent(query=user_query, wardrobe=wardrobe)

    # 4. Handle the error path (e.g., no search results)
    if session.get("error"):
        return session["error"], "", ""

    # 5. Handle the successful path
    # Format the item details nicely for the first UI panel
    item = session["selected_item"]
    listing_text = f"Title: {item.get('title')}\n"
    listing_text += f"Price: ${item.get('price')}\n"
    listing_text += f"Size: {item.get('size')}\n"
    listing_text += f"Condition: {item.get('condition')}\n"
    listing_text += f"Platform: {item.get('platform')}\n"
    listing_text += f"Price Check: {session.get('price_assessment')}\n\n"
    listing_text += f"Description: {item.get('description')}"

    # Append the retry logic message if it exists
    if session.get("retry_message"):
        listing_text += session["retry_message"]
    
    # Return the three strings mapped to the three Gradio output boxes
    return (
        listing_text, 
        session["outfit_suggestion"], 
        session["fit_card"]
    )

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

# ── interface ─────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "vintage graphic tee under $30",
    "90s track jacket in size M",
    "flowy midi skirt under $40",
    "black combat boots size 8",
    "designer ballgown size XXS under $5",   # deliberate no-results test
]

def build_interface():
    with gr.Blocks(title="FitFindr") as demo:
        gr.Markdown("""
# FitFindr 🛍️
Find secondhand pieces and get outfit ideas based on your wardrobe.
Describe what you're looking for — include size and price if you want to filter.
        """)

        with gr.Row():
            query_input = gr.Textbox(
                label="What are you looking for?",
                placeholder="e.g. vintage graphic tee under $30, size M",
                lines=2,
                scale=3,
            )
            wardrobe_choice = gr.Radio(
                choices=["Example wardrobe", "Empty wardrobe (new user)"],
                value="Example wardrobe",
                label="Wardrobe",
                scale=1,
            )

        submit_btn = gr.Button("Find it", variant="primary")

        with gr.Row():
            listing_output = gr.Textbox(
                label="🛍️ Top listing found",
                lines=8,
                interactive=False,
            )
            outfit_output = gr.Textbox(
                label="👗 Outfit idea",
                lines=8,
                interactive=False,
            )
            fitcard_output = gr.Textbox(
                label="✨ Your fit card",
                lines=8,
                interactive=False,
            )

        gr.Examples(
            examples=[[q, "Example wardrobe"] for q in EXAMPLE_QUERIES],
            inputs=[query_input, wardrobe_choice],
            label="Try these queries",
        )

        submit_btn.click(
            fn=handle_query,
            inputs=[query_input, wardrobe_choice],
            outputs=[listing_output, outfit_output, fitcard_output],
        )
        query_input.submit(
            fn=handle_query,
            inputs=[query_input, wardrobe_choice],
            outputs=[listing_output, outfit_output, fitcard_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
