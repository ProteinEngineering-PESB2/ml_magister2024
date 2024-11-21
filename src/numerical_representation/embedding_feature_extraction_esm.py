from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import os
import gc
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df_data = pd.read_csv(sys.argv[1])

name_response = sys.argv[2]
name_sequence = sys.argv[3]
model_name = sys.argv[4]
name_export = sys.argv[5]

print("Processing: ", model_name)
name_model = model_name.split("/")[-1]

tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
model = AutoModel.from_pretrained(model_name).to(device)

embeddings_matrix = []
list_labels = []

for index in df_data.index:
    seq = df_data[name_sequence][index]
    label = df_data[name_response][index]

    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
    
    sequence_embedding = embeddings.mean(dim=1).squeeze().cpu().numpy()
    embeddings_matrix.append(sequence_embedding)
    list_labels.append(label)

header = [f"p_{i+1}" for i in range(len(embeddings_matrix[0]))]

df_embedding = pd.DataFrame(data=embeddings_matrix, columns=header)
df_embedding[name_response] = list_labels

df_embedding.to_csv(name_export, index=False)

#del model
gc.collect()
torch.cuda.empty_cache()
