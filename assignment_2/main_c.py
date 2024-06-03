import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import torch
from openai import OpenAI
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForCausalLM

nlp = spacy.load("de_core_news_lg")


def process_text_file(file_path):
    with open(file_path, "r") as file:
        txt = file.read()
    doc = nlp(txt)
    return doc


def extract_entities(doc):
    entities = {
        "Location": None,
        "Description": None,
        "Date": None,
        "Photographer": None,
        "Film": None
    }

    for ent in doc.ents:
        if ent.label_ == "GPE" or ent.label_ == "LOC":
            if entities["Location"] is None:
                entities["Location"] = ent.text
            else:
                entities["Location"] += " " + ent.text
        elif ent.label_ == "DATE":
            if entities["Date"] is None:
                entities["Date"] = ent.text
            else:
                entities["Date"] += " " + ent.text
        elif ent.label_ == "PERSON" or ent.label_ == "PER":
            if entities["Photographer"] is None:
                entities["Photographer"] = ent.text
            else:
                entities["Photographer"] += " " + ent.text
        elif ent.label_ == "FILM":
            if entities["Film"] is None:
                entities["Film"] = ent.text
            else:
                entities["Film"] += " " + ent.text
        else:
            if entities["Description"] is None:
                entities["Description"] = ent.text
            else:
                entities["Description"] += " " + ent.text

    return entities


def spacy_main(txts_folder):
    patterns = [
        {"label": "DATE", "pattern": [{"TEXT": {
            "REGEX": "^(January|February|March|April|Mai|June|July|August|September|October|November|December|Feber|Nov\\.)$"}},
            {"TEXT": {"REGEX": "^[0-9]{4}$"}}]},
        {"label": "DATE", "pattern": [{"TEXT": {
            "REGEX": "^(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|Feber|Nov\\.)$"}},
            {"TEXT": {"REGEX": "^[0-9]{4}$"}}]},
        {"label": "DATE",
         "pattern": [{"TEXT": {"REGEX": "^(Jan|Feb|Mär|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez|Feber|Nov\\.)$"}},
                     {"TEXT": {"REGEX": "^[0-9]{4}$"}}]},
        {"label": "DATE",
         "pattern": [{"TEXT": {"REGEX": "^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Feber|Nov\\.)$"}},
                     {"TEXT": {"REGEX": "^[0-9]{4}$"}}]},
        {"label": "DATE", "pattern": [{"TEXT": {"REGEX": "^[0-9]{4}$"}}]},
        {"label": "PERSON", "pattern": [{"TEXT": {"REGEX": "^(Aufn)\\.$"}}, {"TEXT": {"REGEX": "^[A-Z][a-z]+$"}}]},
        {"label": "FILM", "pattern": [{"LOWER": "rollfilm"}, {"IS_ASCII": True, "OP": "+"}]},
        {"label": "FILM", "pattern": [{"LOWER": "film"}, {"IS_ASCII": True, "OP": "+"}]}
    ]

    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(patterns)

    # txts_folder = "ab_output"
    all_vectors = []
    all_labels = []
    entity_vectors = {}
    similar_words_dataset = {}

    for txt_file in os.listdir(txts_folder):
        file_path = os.path.join(txts_folder, txt_file)
        doc = process_text_file(file_path)
        print(f"Processing {txt_file}:")

        for token in doc:
            if token.has_vector:
                all_vectors.append(token.vector)
                all_labels.append(token.text)
                similar_words = [w.text for w in doc if w.similarity(token) > 0.3 and w != token]
                if similar_words:
                    if token.text not in similar_words_dataset:
                        similar_words_dataset[token.text] = set()
                    similar_words_dataset[token.text].update(similar_words)

        for ent in doc.ents:
            if ent.label_ not in entity_vectors:
                entity_vectors[ent.label_] = {"vectors": [], "labels": []}

            for token in ent:
                if token.has_vector:
                    entity_vectors[ent.label_]["vectors"].append(token.vector)
                    entity_vectors[ent.label_]["labels"].append(token.text)

        entities = extract_entities(doc)
        print(entities)

        if not os.path.exists("spacy_output"):
            os.makedirs("spacy_output")
        json_file_path = os.path.join("spacy_output", f"{os.path.splitext(txt_file)[0]}.json")
        with open(json_file_path, "w") as json_file:
            json.dump(entities, json_file, ensure_ascii=False, indent=4)

    all_vectors = np.array(all_vectors)
    all_labels = np.array(all_labels)

    # Visualize word vectors using t-SNE
    perplexity_value = min(30, all_vectors.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_value)
    transformed = tsne.fit_transform(all_vectors)

    plt.figure(figsize=(20, 12))
    plt.scatter(transformed[:, 0], transformed[:, 1])

    for i, label in enumerate(all_labels):
        plt.annotate(label, (transformed[i, 0], transformed[i, 1]))

    plt.show()

    for entity, data in entity_vectors.items():
        vectors = np.array(data["vectors"])
        labels = np.array(data["labels"])

        if vectors.shape[0] > 0:  # Ensure there are vectors to process
            perplexity_value = min(30, vectors.shape[0] - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity_value)
            transformed = tsne.fit_transform(vectors)

            plt.figure(figsize=(20, 12))
            plt.scatter(transformed[:, 0], transformed[:, 1])

            for i, label in enumerate(labels):
                plt.annotate(label, (transformed[i, 0], transformed[i, 1]))

            plt.title(f't-SNE visualization for entity type: {entity}')
            plt.show()
        else:
            print(f"No vectors found for entity type: {entity}")

    df = pd.DataFrame([(word, ', '.join(similars)) for word, similars in similar_words_dataset.items()],
                      columns=["Word", "Similar_Words"])

    df.to_csv("similar_words_dataset.csv", index=False)
    print(df)


def read_text_file(file_path):
    with open(file_path, "r") as file:
        txt = file.read()
    return txt


def llm_local(txts_folder):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    def generate_response(messages):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            inputs,
            # attention_mask=inputs['attention_mask'],
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        response = outputs[0][inputs.shape[-1]:]
        return tokenizer.decode(response, skip_special_tokens=True)

    # txts_folder = "ab_output"
    for txt_file in os.listdir(txts_folder):
        file_path = os.path.join(txts_folder, txt_file)
        txt = read_text_file(file_path)
        print(f"Processing {txt_file}:")

        messages = [
            {
                "role": "system", "content":
                '''
                Answer strictly in the form of a dictionary with the following keys:
                {
                    "Location": "Helenental",
                    "Description": "unerlaubte Rodung im Schutzgebiet",
                    "Date": "April 1948",
                    "Photographer": "Meisinger",
                    "Film": "Neg.Nr. 3254/KIX/16, Film",
                }
                '''
            },
        ]
        new_message = {"role": "user", "content": txt}
        print(txt)
        messages.append(new_message)

        follow_up_response = generate_response(messages)
        print(follow_up_response)

        try:
            follow_up_response_dict = json.loads(follow_up_response)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            follow_up_response_dict = {"error": "Failed to decode response"}

        if not os.path.exists("llama_output"):
            os.makedirs("llama_output")

        json_file_path = os.path.join("llama_output", f"{os.path.splitext(txt_file)[0]}.json")
        with open(json_file_path, "w") as json_file:
            json.dump(follow_up_response_dict, json_file, ensure_ascii=False, indent=4)


def llm_api(txts_folder, api_key):
    openai = OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    # txts_folder = "ab_output"
    for txt_file in os.listdir(txts_folder):
        file_path = os.path.join(txts_folder, txt_file)
        txt = read_text_file(file_path)
        print(f"Processing {txt_file}:")

        messages = [
            {
                "role": "system", "content":
                '''
                Answer strictly in the form of a dictionary with the following keys:
                {
                    "Location": "Helenental",
                    "Description": "unerlaubte Rodung im Schutzgebiet",
                    "Date": "April 1948",
                    "Photographer": "Meisinger",
                    "Film": "Neg.Nr. 3254/KIX/16, Film",
                }
                '''
            },
        ]

        new_message = {"role": "user", "content": txt}
        print(txt)
        messages.append(new_message)

        chat_completion = openai.chat.completions.create(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            messages=messages,
        )

        follow_up_response = chat_completion.choices[0].message.content
        print(follow_up_response)

        try:
            follow_up_response_dict = json.loads(follow_up_response.replace(',\n}', '\n}'))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            follow_up_response_dict = {"error": "Failed to decode response"}

        if not os.path.exists("llama_output"):
            os.makedirs("llama_output")

        json_file_path = os.path.join("llama_output", f"{os.path.splitext(txt_file)[0]}.json")
        with open(json_file_path, "w") as json_file:
            json.dump(follow_up_response_dict, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--txts_folder", type=str, default="ab_output",
                        help="Path to the folder containing the text files.")
    parser.add_argument("--llm_local", type=bool, default=False, help="Whether to run the LLM model.")
    parser.add_argument("--llm_api", type=bool, default=False, help="Whether to run the LLM API.")
    parser.add_argument("--api_key", type=str, default="", help="API key for the LLM API.")
    parser.add_argument("--spacy", type=bool, default=True, help="Whether to run the Spacy model.")
    args = parser.parse_args()

    if args.spacy:
        spacy_main(args.txts_folder)
    if args.llm_api and args.api_key:
        llm_api(args.txts_folder, args.api_key)
    if args.llm_local:
        llm_local(args.txts_folder)
