import pickle

from src.model import TransformersCRF
import torch
from termcolor import colored
from src.data import TransformersNERDataset
from typing import List
from transformers import AutoTokenizer
import tarfile
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


class TransformersNERPredictor:
    def __init__(self, model_archived_file: str, cuda_device: str = "cpu"):
        """
        model_archived_file: ends with "tar.gz"
        OR
        directly use the model folder patth
        """
        device = torch.device(cuda_device)
        if model_archived_file.endswith("tar.gz"):
            tar = tarfile.open(model_archived_file)
            self.conf = pickle.load(tar.extractfile(tar.getnames()[1]))  ## config file
            self.model = TransformersCRF(self.conf)
            self.model.load_state_dict(
                torch.load(tar.extractfile(tar.getnames()[2]), map_location=device)
            )  ## model file
        else:
            folder_name = model_archived_file
            assert os.path.isdir(folder_name)
            f = open(folder_name + "/config.conf", "rb")
            self.conf = pickle.load(f)
            f.close()
            self.model = TransformersCRF(self.conf)
            self.model.load_state_dict(
                torch.load(f"{folder_name}/lstm_crf.m", map_location=device)
            )
        self.conf.device = device
        self.model.to(device)
        self.model.eval()

        print(
            colored(
                f"[Data Info] Tokenizing the instances using '{self.conf.embedder_type}' tokenizer",
                "blue",
            )
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf.embedder_type)

    def predict(self, sents: List[List[str]], batch_size=-1):
        batch_size = len(sents) if batch_size == -1 else batch_size

        dataset = TransformersNERDataset(
            file=None,
            sents=sents,
            tokenizer=self.tokenizer,
            label2idx=self.conf.label2idx,
            is_train=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=dataset.collate_fn,
        )

        all_predictions = []
        for batch_id, batch in tqdm(
            enumerate(loader, 0), desc="--evaluating batch", total=len(loader)
        ):
            one_batch_insts = dataset.insts[
                batch_id * batch_size : (batch_id + 1) * batch_size
            ]
            batch_max_scores, batch_max_ids = self.model(
                subword_input_ids=batch.input_ids.to(self.conf.device),
                word_seq_lens=batch.word_seq_len.to(self.conf.device),
                orig_to_tok_index=batch.orig_to_tok_index.to(self.conf.device),
                attention_mask=batch.attention_mask.to(self.conf.device),
                is_train=False,
            )

            for idx in range(len(batch_max_ids)):
                length = batch.word_seq_len[idx]
                prediction = batch_max_ids[idx][:length].tolist()
                prediction = prediction[::-1]
                prediction = [self.conf.idx2labels[p] for p in prediction]
                one_batch_insts[idx].prediction = prediction
                all_predictions.append(prediction)
        return all_predictions

    def predict_and_display(self, sents: List[List[str]], batch_size=-1):
        predictions = self.predict(sents, batch_size)

        prefix_map = {"B-": "[BEGIN] ", "I-": "[INSIDE]", "E-": "[END]   "}

        for i, (sent, pred) in enumerate(zip(sents, predictions)):
            sentence_text = " ".join(sent)
            print(f"Sentence {i + 1}: {sentence_text}")
            print("-" * 75)

            if len(sent) != len(pred):
                print(
                    f"Warning: Mismatch between words ({len(sent)}) and predictions ({len(pred)})"
                )
                min_len = min(len(sent), len(pred))
                sent = sent[:min_len]
                pred = pred[:min_len]

            for j, (word, label) in enumerate(zip(sent, pred)):
                label_prefix = "[OTHER] "
                entity_type = ""

                for prefix, prefix_str in prefix_map.items():
                    if label.startswith(prefix):
                        label_prefix = prefix_str
                        entity_type = label[len(prefix) :]
                        break

                if entity_type:
                    display_label = f"{label_prefix} {entity_type}"
                else:
                    display_label = f"{label_prefix} {label}"

                print(f"  {j + 1:2d}. {word:20} -> {display_label}")

        return predictions

    def predict_and_display_with_entities(self, sents: List[List[str]], batch_size=-1):
        predictions = self.predict(sents, batch_size)
        all_entities = []

        prefix_map = {"B-": "[BEGIN] ", "I-": "[INSIDE]", "E-": "[END]   "}

        for i, (sent, pred) in enumerate(zip(sents, predictions)):
            sentence_text = " ".join(sent)
            print(f"Sentence {i + 1}: {sentence_text}")
            print("-" * 75)

            if len(sent) != len(pred):
                print(
                    f"Warning: Mismatch between words ({len(sent)}) and predictions ({len(pred)})"
                )
                min_len = min(len(sent), len(pred))
                sent = sent[:min_len]
                pred = pred[:min_len]

            for j, (word, label) in enumerate(zip(sent, pred)):
                label_prefix = "[OTHER] "
                entity_type = ""

                for prefix, prefix_label in prefix_map.items():
                    if label.startswith(prefix):
                        label_prefix = prefix_label
                        entity_type = label[len(prefix) :]
                        break

                if entity_type:
                    display_label = f"{label_prefix} {entity_type}"
                else:
                    display_label = f"{label_prefix} {label}"

                print(f"  {j + 1:2d}. {word:20} -> {display_label}")

            entities = self._extract_entities(sent, pred)
            all_entities.append(entities)

            if entities:
                print("\nExtracted Entities:")
                print("-" * 40)
                for entity_type, entity_list in entities.items():
                    for entity_text, start_idx, end_idx in entity_list:
                        print(
                            f"  {entity_type}: '{entity_text}' (position {start_idx}-{end_idx})"
                        )
            else:
                print("\nNo entities found.")

        return {"predictions": predictions, "entities": all_entities}

    def _extract_entities(self, words: List[str], labels: List[str]) -> dict:
        """
        Extract entities from word-label pairs

        Args:
            words: List of words
            labels: List of corresponding labels

        Returns:
            dict: Dictionary mapping entity types to list of (text, start_idx, end_idx) tuples
        """
        entities = {}
        current_entity = None
        current_words = []
        start_idx = -1

        for i, (word, label) in enumerate(zip(words, labels)):
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity and current_words:
                    entity_text = " ".join(current_words)
                    if current_entity not in entities:
                        entities[current_entity] = []
                    entities[current_entity].append((entity_text, start_idx, i - 1))

                # Start new entity
                current_entity = label[2:]
                current_words = [word]
                start_idx = i

            elif label.startswith("I-") and current_entity == label[2:]:
                # Continue current entity
                current_words.append(word)

            elif label.startswith("E-") and current_entity == label[2:]:
                # End current entity
                current_words.append(word)
                entity_text = " ".join(current_words)
                if current_entity not in entities:
                    entities[current_entity] = []
                entities[current_entity].append((entity_text, start_idx, i))

                # Reset
                current_entity = None
                current_words = []
                start_idx = -1

            else:
                # Save previous entity if exists and reset
                if current_entity and current_words:
                    entity_text = " ".join(current_words)
                    if current_entity not in entities:
                        entities[current_entity] = []
                    entities[current_entity].append((entity_text, start_idx, i - 1))

                current_entity = None
                current_words = []
                start_idx = -1

        # Handle entity that extends to end of sentence
        if current_entity and current_words:
            entity_text = " ".join(current_words)
            if current_entity not in entities:
                entities[current_entity] = []
            entities[current_entity].append((entity_text, start_idx, len(words) - 1))

        return entities


if __name__ == "__main__":
    sents = [
        [
            "I",
            "am",
            "traveling",
            "to",
            "Singapore",
            "to",
            "visit",
            "the",
            "Merlion",
            "Park",
            ".",
        ],
        ["John", "cannot", "come", "with", "us", "."],
    ]
    model_path = "model_files/english_model"
    device = "cpu"  # cpu, cuda:0, cuda:1
    ## or model_path = "model_files/english_model.tar.gz"
    predictor = TransformersNERPredictor(model_path, cuda_device=device)
    prediction = predictor.predict(sents)
    print(prediction)
