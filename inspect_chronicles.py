from datasets import load_dataset
try:
    dataset = load_dataset("jihyoung/ConversationChronicles", split="train", streaming=True)
    row = next(iter(dataset))
    print(row.keys())
    print(row)
except Exception as e:
    print("Error:", e)
