import json
import re

import panel as pn

from src.llm_order_updated import get_completion_from_messages

pn.extension()

# The menu is the source of truth for availability and pricing.
MENU = {
    "pizza": {
        "pepperoni": {"large": 12.95, "medium": 10.00, "small": 7.00},
        "cheese": {"large": 10.95, "medium": 9.25, "small": 6.50},
        "eggplant": {"large": 11.95, "medium": 9.75, "small": 6.75},
    },
    "toppings": {
        "extra cheese": 2.00,
        "mushrooms": 1.50,
        "sausage": 3.00,
        "canadian bacon": 3.50,
        "ai sauce": 1.50,
        "peppers": 1.00,
    },
    "drinks": {
        "coke": {"large": 3.00, "medium": 2.00, "small": 1.00},
        "sprite": {"large": 3.00, "medium": 2.00, "small": 1.00},
        "bottled water": {"regular": 5.00},
    },
    "sides": {
        "fries": {"large": 4.50, "small": 3.50},
        "greek salad": {"regular": 7.25},
    },
}

SYSTEM_PROMPT = """
You are OrderBot, a friendly pizza restaurant ordering assistant.

First greet the customer and display the menu. Collect the customer's order
conversationally. Clarify item sizes and toppings. Do not accept items that
are not on the menu. Ask whether the order is pickup or delivery. For delivery,
collect the address. Summarize the order and ask whether the customer wants
anything else before confirming it.

Menu:
Pizzas:
- Pepperoni: large $12.95, medium $10.00, small $7.00
- Cheese: large $10.95, medium $9.25, small $6.50
- Eggplant: large $11.95, medium $9.75, small $6.75

Additional toppings:
- Extra cheese $2.00
- Mushrooms $1.50
- Sausage $3.00
- Canadian bacon $3.50
- AI sauce $1.50
- Peppers $1.00

Drinks:
- Coke: large $3.00, medium $2.00, small $1.00
- Sprite: large $3.00, medium $2.00, small $1.00
- Bottled water: regular $5.00

Sides:
- Fries: large $4.50, small $3.50
- Greek salad: regular $7.25

Remember the conversation so far. Be concise and conversational.
Do not calculate or promise a final total; the application calculates prices.
"""

# This list is the conversation memory. Each turn is appended and sent to the LLM.
context = [{"role": "system", "content": SYSTEM_PROMPT}]
panels = []


def collect_messages(_):
    """Send the customer's message and display the assistant's reply."""
    prompt = inp.value.strip()

    if not prompt:
        status.object = "Please enter a message before clicking Chat."
        return pn.Column(*panels)

    inp.value = ""
    status.object = ""

    context.append({"role": "user", "content": prompt})

    try:
        response = get_completion_from_messages(context, temperature=0.4)
    except Exception as exc:
        # Do not append a failed assistant response to conversation history.
        context.pop()
        status.object = f"Could not contact the language model: {exc}"
        return pn.Column(*panels)

    context.append({"role": "assistant", "content": response})

    panels.append(pn.Row("User:", pn.pane.Markdown(prompt, width=600)))
    panels.append(
        pn.Row(
            "Assistant:",
            pn.pane.Markdown(
                response,
                width=600,
                styles={"background-color": "#F6F6F6"},
            ),
        )
    )
    return pn.Column(*panels)


def extract_json(text):
    """Parse JSON, including when the model wraps it in a Markdown code fence."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Recover a JSON object if the model added a short sentence around it.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("The model did not return a valid JSON object.")
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("The model returned invalid JSON. Please try again.") from exc


def validate_order(order):
    """Check all order items and sizes against MENU before calculating prices."""
    if not isinstance(order, dict):
        return False, "The order must be a JSON object."

    pizza = order.get("pizza")
    if not isinstance(pizza, dict):
        return False, "I couldn't identify a pizza in the order."

    pizza_type = str(pizza.get("type", "")).strip().lower()
    pizza_size = str(pizza.get("size", "")).strip().lower()

    if pizza_type not in MENU["pizza"]:
        return False, f"'{pizza_type or 'Missing pizza'}' is not on the pizza menu."
    if pizza_size not in MENU["pizza"][pizza_type]:
        return False, f"'{pizza_size or 'Missing size'}' is not a valid size for {pizza_type} pizza."

    toppings = pizza.get("toppings", [])
    if not isinstance(toppings, list):
        return False, "Pizza toppings must be provided as a list."
    for topping in toppings:
        if str(topping).strip().lower() not in MENU["toppings"]:
            return False, f"'{topping}' is not an available additional topping."

    for section in ("drinks", "sides"):
        items = order.get(section, [])
        if not isinstance(items, list):
            return False, f"{section.capitalize()} must be provided as a list."

        for item in items:
            if not isinstance(item, dict):
                return False, f"Each {section[:-1]} must be an object."
            name = str(item.get("type", "")).strip().lower()
            size = str(item.get("size", "")).strip().lower()

            if name not in MENU[section]:
                return False, f"'{name or 'Unknown item'}' is not on the {section} menu."
            if size not in MENU[section][name]:
                valid_sizes = ", ".join(MENU[section][name].keys())
                return False, f"Invalid size for {name}. Available size(s): {valid_sizes}."

    return True, "Order validated successfully."


def price_order(order):
    """Build a priced order and calculate the total using Python, not the LLM."""
    pizza = order["pizza"]
    pizza_type = pizza["type"].strip().lower()
    pizza_size = pizza["size"].strip().lower()

    base_price = MENU["pizza"][pizza_type][pizza_size]
    priced_toppings = []
    for topping in pizza.get("toppings", []):
        topping_name = topping.strip().lower()
        priced_toppings.append(
            {"name": topping_name, "price": MENU["toppings"][topping_name]}
        )

    pizza_total = base_price + sum(item["price"] for item in priced_toppings)
    priced_order = {
        "pizza": {
            "type": pizza_type,
            "size": pizza_size,
            "base_price": round(base_price, 2),
            "toppings": priced_toppings,
            "line_total": round(pizza_total, 2),
        },
        "drinks": [],
        "sides": [],
    }

    total = pizza_total

    for drink in order.get("drinks", []):
        name = drink["type"].strip().lower()
        size = drink["size"].strip().lower()
        price = MENU["drinks"][name][size]
        priced_order["drinks"].append(
            {"type": name, "size": size, "price": round(price, 2)}
        )
        total += price

    for side in order.get("sides", []):
        name = side["type"].strip().lower()
        size = side["size"].strip().lower()
        price = MENU["sides"][name][size]
        priced_order["sides"].append(
            {"type": name, "size": size, "price": round(price, 2)}
        )
        total += price

    priced_order["total_price"] = round(total, 2)
    return priced_order


def generate_order_summary(_):
    """Ask the LLM to extract order details; validate and price them in Python."""
    summary_output.objects = []
    summary_status.object = "Preparing your order summary..."

    extraction_prompt = """
Extract the customer's CURRENT order from the conversation.
Return ONLY valid JSON, with no Markdown and no explanatory text.
Use exactly this structure:
{
  "pizza": {
    "type": "pepperoni",
    "size": "large",
    "toppings": ["extra cheese"]
  },
  "drinks": [
    {"type": "coke", "size": "medium"}
  ],
  "sides": [
    {"type": "fries", "size": "large"}
  ]
}
Rules:
- "pizza.type" must be the base pizza: pepperoni, cheese, or eggplant.
- "pizza.toppings" must contain only additional toppings, not the base pizza name.
- Use empty arrays when there are no toppings, drinks, or sides.
- Do not invent items, sizes, quantities, or prices.
- If a required pizza type or size is missing or unclear, use an empty string.
- Do not include a price or total; the application calculates prices.
"""

    # Keep the original conversation unchanged; use a separate copy for extraction.
    summary_messages = context.copy()
    summary_messages.append({"role": "user", "content": extraction_prompt})

    try:
        raw_summary = get_completion_from_messages(summary_messages, temperature=0)
        order = extract_json(raw_summary)
        is_valid, validation_message = validate_order(order)

        if not is_valid:
            summary_status.object = (
                f"**Order needs attention:** {validation_message} "
                "Please clarify the order in the chat, then generate the summary again."
            )
            return

        priced_order = price_order(order)
        summary_output.objects = [
            pn.pane.Markdown(
                "### Verified order summary\n"
                "```json\n"
                + json.dumps(priced_order, indent=2)
                + "\n```"
            )
        ]
        summary_status.object = "Order validated. Prices and total were calculated from the menu."

    except Exception as exc:
        summary_status.object = f"Could not generate the order summary: {exc}"


# Panel widgets
inp = pn.widgets.TextInput(value="Hi", placeholder="Enter your order here...")
button_conversation = pn.widgets.Button(name="Chat!", button_type="primary")
summary_button = pn.widgets.Button(name="Generate Order Summary", button_type="success")

chat_display = pn.bind(collect_messages, button_conversation)
summary_button.on_click(generate_order_summary)

status = pn.pane.Markdown("")
summary_status = pn.pane.Markdown("")
summary_output = pn.Column()

dashboard = pn.Column(
    pn.pane.Markdown("# OrderBot - Pizza Restaurant"),
    pn.pane.Markdown("Chat with OrderBot to place your order."),
    inp,
    button_conversation,
    status,
    pn.panel(chat_display, loading_indicator=True, height=350),
    pn.layout.Divider(),
    summary_button,
    summary_status,
    summary_output,
)

dashboard.servable()
