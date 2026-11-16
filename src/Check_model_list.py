import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GPT_LLM_API_KEY")
)

models = client.models.list()

small = sorted(m.id for m in models if any(k in m.id.lower() for k in ("nano", "mini")))

print(f"Small models ({len(small)}):")
print("-" * 40)
for i, model_id in enumerate(small, 1):
    print(f"{i:>2}. {model_id}")
# for model in models.data:
#     print(model.id)