import os

from src.llm_order import get_completion, get_completion_from_messages  # noqa: E402

def collect_messages(_):
    prompt = inp.value.strip()
    if not prompt:
        return pn.Column(*panels)
    inp.value = ''
    context.append({'role':'user', 'content':f"{prompt}"})
    response = get_completion_from_messages(context) 
    context.append({'role':'assistant', 'content':f"{response}"})

    panels.append(
        pn.Row('User:', pn.pane.Markdown(prompt, width=600)))
    panels.append(
        pn.Row('Assistant:', pn.pane.Markdown(response, width=600, styles={'background-color': '#F6F6F6'})))
 
    return pn.Column(*panels)

import panel as pn  # GUI
pn.extension()

panels = [] # collect display 

context = [ {'role':'system', 'content':"""
You are OrderBot, an automated service to display and collect orders for a pizza restaurant. \
You first greet the customer and display the menu, then collect the order, \
and then asks if it's a pickup or delivery. \
Deny the item not present in the menu when customer asks for or orders it.\
You wait to collect the entire order, then summarize it and check for a final \
time if the customer wants to add anything else. \
If it's a delivery, you ask for an address. \
Finally you collect the payment.\
Make sure to clarify all options, extras and sizes to uniquely \
identify the item from the menu.\
You respond in a short, very conversational friendly style. \
The menu includes \
pepperoni pizza  12.95, 10.00, 7.00 \
cheese pizza   10.95, 9.25, 6.50 \
eggplant pizza   11.95, 9.75, 6.75 \
fries 4.50, 3.50 \
greek salad 7.25 \
Toppings: \
extra cheese 2.00, \
mushrooms 1.50 \
sausage 3.00 \
canadian bacon 3.50 \
AI sauce 1.50 \
peppers 1.00 \
Drinks: \
coke 3.00, 2.00, 1.00 \
sprite 3.00, 2.00, 1.00 \
bottled water 5.00 \
"""} ]  # accumulate messages


MENU = {
    "pizza": {
        "pepperoni": {"large": 12.95, "medium": 10.00, "small": 7.00},
        "cheese": {"large": 10.95, "medium": 9.25, "small": 6.50},
        "eggplant": {"large": 11.95, "medium": 9.75, "small": 6.75}
    },
    "toppings": {
        "extra cheese": 2.00,
        "mushrooms": 1.50,
        "sausage": 3.00,
        "canadian bacon": 3.50,
        "ai sauce": 1.50,
        "peppers": 1.00
    },
    "drinks": {
        "coke": {"large": 3.00, "medium": 2.00, "small": 1.00},
        "sprite": {"large": 3.00, "medium": 2.00, "small": 1.00},
        "bottled water": {"regular": 5.00}
    },
    "sides": {
        "fries": {"large": 4.50, "small": 3.50},
        "greek salad": {"regular": 7.25}
    }
}

def validate_pizza(pizza, size):
    if pizza not in MENU["pizza"]:
        return False, "Sorry, that pizza is not on our menu."

    if size not in MENU["pizza"][pizza]:
        return False, "Please select small, medium or large."

    return True, "Valid pizza selection."

def calculate_order_total(order):
    total = 0.0

    # Pizza price
    pizza = order["pizza"]["type"]
    size = order["pizza"]["size"]

    total += MENU["pizza"][pizza][size]

    # Additional toppings
    for topping in order["pizza"].get("toppings", []):
        total += MENU["toppings"][topping]

    # Drinks
    for drink in order.get("drinks", []):
        name = drink["type"]
        size = drink["size"]

        total += MENU["drinks"][name][size]

    # Sides
    for side in order.get("sides", []):
        name = side["type"]
        size = side["size"]

        total += MENU["sides"][name][size]

    return round(total, 2)

inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here…')
button_conversation = pn.widgets.Button(name="Chat!")

interactive_conversation = pn.bind(collect_messages, button_conversation)

dashboard = pn.Column(
    inp,
    pn.Row(button_conversation),
    pn.panel(interactive_conversation, loading_indicator=True, height=300),
)



messages =  context.copy()
messages.append(
{'role':'system', 'content':'create a json summary of the previous food order. Itemize the price for each item\
 The fields should be 1) pizza, include size 2) list of toppings 3) list of drinks, include size   4) list of sides include size  5)total price '},    
)
# messages.append({
#     'role': 'system',
#     'content': """
#     Return ONLY valid JSON using this structure:

#     {
#       "pizza": {
#         "type": "pepperoni",
#         "size": "large",
#         "toppings": ["extra cheese"]
#       },
#       "drinks": [
#         {"type": "coke", "size": "medium"}
#       ],
#       "sides": [
#         {"type": "fries", "size": "large"}
#       ]
#     }

#     Do not include prices or total_price.
#     Use empty arrays if there are no toppings, drinks or sides.
#     """
# })
 #The fields should be 1) pizza, price 2) list of toppings 3) list of drinks, include size include price  4) list of sides include size include price, 5)total price '},    

response = get_completion_from_messages(messages, temperature=0)
print(response)
dashboard.servable()