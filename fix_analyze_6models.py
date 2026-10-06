# Updates analyze_results.py MODEL_LABELS to remove gemma2_9b (6-model study)
content = open("analyze_results.py", encoding="utf-8").read()

old = '''MODEL_LABELS = {
    "llama_3_3_70b": "Llama 3.3 70B",
    "llama_4_scout":  "Llama 4 Scout",
    "qwen3_32b":      "Qwen3 32B",
    "deepseek_r1":    "DeepSeek R1",
    "mixtral_8x7b":   "Mixtral 8x7B",
    "llama_3_1_8b":   "Llama 3.1 8B",
    "gemma2_9b":      "Gemma2 9B",
}'''

new = '''MODEL_LABELS = {
    "llama_3_3_70b": "Llama 3.3 70B",
    "llama_4_scout":  "Llama 4 Scout",
    "qwen3_32b":      "Qwen3 32B",
    "deepseek_r1":    "DeepSeek R1",
    "mixtral_8x7b":   "Mixtral 8x7B",
    "llama_3_1_8b":   "Llama 3.1 8B",
}'''

if old in content:
    content = content.replace(old, new)
    open("analyze_results.py", "w", encoding="utf-8").write(content)
    print("Updated to 6-model study - gemma2_9b removed from MODEL_LABELS")
else:
    print("Pattern not found exactly - checking current content...")
    import re
    m = re.search(r'MODEL_LABELS = \{[^}]+\}', content)
    if m:
        print(m.group())

# Also update COLORS list to have 6 entries instead of 7
old_colors = 'COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]'
new_colors = 'COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]'
content2 = open("analyze_results.py", encoding="utf-8").read()
if old_colors in content2:
    content2 = content2.replace(old_colors, new_colors)
    open("analyze_results.py", "w", encoding="utf-8").write(content2)
    print("Updated COLORS list to 6 entries")
