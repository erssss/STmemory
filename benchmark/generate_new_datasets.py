import json
import os
from datetime import datetime, timedelta
from datasets import load_dataset

def generate_soda(output_path, num_samples=10):
    print("Loading SODA dataset...")
    ds = load_dataset("allenai/soda", split="train", streaming=True)
    results = []
    
    for idx, row in enumerate(ds):
        if idx >= num_samples:
            break
            
        dialogue = row.get("dialogue", [])
        speakers = row.get("speakers", [])
        narrative = row.get("narrative", "")
        
        if not dialogue or not speakers:
            continue
            
        unique_speakers = list(dict.fromkeys(speakers))
        speaker_a = unique_speakers[0] if len(unique_speakers) > 0 else "Person A"
        speaker_b = unique_speakers[1] if len(unique_speakers) > 1 else "Person B"
        
        session_1 = []
        for s, text in zip(speakers, dialogue):
            session_1.append({
                "speaker": s,
                "text": text
            })
            
        item = {
            "qa": [
                {
                    "question": "What is the main narrative or summary of this conversation?",
                    "answer": narrative,
                    "category": 1,
                    "evidence": []
                }
            ],
            "conversation": {
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "session_1": session_1,
                "session_1_date_time": "2024-01-01 10:00:00"
            }
        }
        results.append(item)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} SODA examples to {output_path}")

def generate_chronicles(output_path, num_samples=10):
    print("Loading ConversationChronicles dataset...")
    ds = load_dataset("jihyoung/ConversationChronicles", split="train", streaming=True)
    results = []
    
    for idx, row in enumerate(ds):
        if idx >= num_samples:
            break
            
        relationship = row.get("relationship", "")
        summaries = row.get("summary", [])
        
        session_keys = ["first", "second", "third", "fourth", "fifth"]
        
        speaker_a = "Person A"
        speaker_b = "Person B"
        
        # Try to find actual speakers from the first available session
        for sk in session_keys:
            spks = row.get(f"{sk}_session_speakers", [])
            if spks:
                unique = list(dict.fromkeys(spks))
                if len(unique) > 0: speaker_a = unique[0]
                if len(unique) > 1: speaker_b = unique[1]
                break
                
        conversation = {
            "speaker_a": speaker_a,
            "speaker_b": speaker_b
        }
        
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        qa_list = [
            {
                "question": "What is the relationship between the speakers?",
                "answer": relationship,
                "category": 1,
                "evidence": []
            }
        ]
        
        for i, sk in enumerate(session_keys):
            dial = row.get(f"{sk}_session_dialogue", [])
            spks = row.get(f"{sk}_session_speakers", [])
            
            if not dial or not spks:
                continue
                
            session_data = []
            for s, text in zip(spks, dial):
                session_data.append({
                    "speaker": s,
                    "text": text
                })
                
            session_id = f"session_{i+1}"
            conversation[session_id] = session_data
            
            current_time = base_time + timedelta(days=i)
            conversation[f"{session_id}_date_time"] = current_time.strftime("%Y-%m-%d %H:%M:%S")
            
            if i < len(summaries):
                qa_list.append({
                    "question": f"Summarize session {i+1} of the conversation.",
                    "answer": summaries[i],
                    "category": 1,
                    "evidence": []
                })
                
        item = {
            "qa": qa_list,
            "conversation": conversation
        }
        results.append(item)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} Chronicles examples to {output_path}")

if __name__ == "__main__":
    generate_soda("soda/dataset/soda10.json", num_samples=10)
    generate_chronicles("chronicles/dataset/chronicles10.json", num_samples=10)

