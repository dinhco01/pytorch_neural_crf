import pickle

from src.model import NNCRF
import torch

from src.config import ContextEmb
from typing import List, Union, Dict
import tarfile

from preprocess.get_elmo_vec import load_elmo, parse_sentence
from src.data import Instance
from src.data.data_utils import UNK
from torch.utils.data._utils.collate import default_collate
import collections
import numpy as np
from src.data.data_utils import check_all_obj_is_None
from tqdm import tqdm

"""
Predictor usage example:
NOTE: this is only used by glove/elmo

    sentence = "This is a sentence"
    # Or you can make a list of sentence:
    # sentence = ["This is a sentence", "This is the second sentence"]

    model_path = "english_model.tar.gz"
    predictor = NERPredictor("model_files/english_model.tar.gz")
    res = predictor.predict(sentence)
    print(res)

"""

Feature = collections.namedtuple(
    "Feature", "words word_seq_len context_emb chars char_seq_lens labels"
)
Feature.__new__.__defaults__ = (None,) * 6


class NERPredictor:
    """
    Usage: for models using word embedding.
    sentence = "This is a sentence"
    model_path = "model_files.tar.gz"
    model = Predictor(model_path)
    prediction = model.predict(sentence)
    """

    def __init__(self, model_archived_file: str, cuda_device: str = "cpu"):
        tar = tarfile.open(model_archived_file)
        tar.extractall()
        folder_name = tar.getnames()[0]
        tar.close()

        f = open(folder_name + "/config.conf", "rb")
        self.conf = pickle.load(f)  # variables come out in the order you put them in
        # default batch size for conf is `10`
        f.close()
        device = torch.device(cuda_device)
        self.conf.device = device
        self.model = NNCRF(self.conf)
        self.model.load_state_dict(
            torch.load(folder_name + "/lstm_crf.m", map_location=device)
        )
        self.model.eval()

        if self.conf.static_context_emb != ContextEmb.none:
            if cuda_device == "cpu":
                cuda_device = -1
            else:
                cuda_device = int(cuda_device.split(":")[1])
            self.elmo = load_elmo(cuda_device)

    def predict_insts(self, batch: Feature) -> List[List[str]]:
        batch_max_scores, batch_max_ids = self.model.decode(
            words=batch.words.to(self.conf.device),
            word_seq_lens=batch.word_seq_len.to(self.conf.device),
            context_emb=batch.context_emb.to(self.conf.device)
            if batch.context_emb is not None
            else None,
            chars=batch.chars.to(self.conf.device),
            char_seq_lens=batch.char_seq_lens.to(self.conf.device),
        )
        predictions = []
        for idx in range(len(batch_max_ids)):
            length = batch.word_seq_len[idx]
            prediction = batch_max_ids[idx][:length].tolist()
            prediction = prediction[::-1]  ## reverse the Viterbi sequence
            prediction = [self.conf.idx2labels[p] for p in prediction]
            predictions.append(prediction)
        return predictions

    def sent_to_insts(self, sentence: str) -> List[Instance]:
        words = sentence.split()
        return [Instance(words=words, ori_words=words)]

    def sents_to_insts(self, sentences: List[str]) -> List[Instance]:
        import nltk

        nltk.download("punkt_tab")

        insts = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            insts.append(Instance(words=words, ori_words=words))
        return insts

    def create_batch_data(self, insts: List[Instance]):
        inst_ids = []
        for inst in insts:
            words = inst.words
            word_ids = []
            char_ids = []
            char_seq_lens = []
            for word in words:
                if word in self.conf.word2idx:
                    word_ids.append(self.conf.word2idx[word])
                else:
                    word_ids.append(self.conf.word2idx[UNK])
                char_id = []
                char_seq_lens.append(len(word))
                for c in word:
                    if c in self.conf.char2idx:
                        char_id.append(self.conf.char2idx[c])
                    else:
                        char_id.append(self.conf.char2idx[UNK])
                char_ids.append(char_id)
            inst_ids.append(
                Feature(
                    words=word_ids,
                    chars=char_ids,
                    word_seq_len=len(words),
                    char_seq_lens=char_seq_lens,
                    context_emb=inst.elmo_vec if hasattr(inst, "elmo_vec") else None,
                    labels=None,
                )
            )

        return self._create_batch_data(inst_ids)

    def _create_batch_data(self, insts: List[Feature]):
        word_seq_lens = [len(feature.words) for feature in insts]
        max_seq_len = max(word_seq_lens)
        max_char_seq_len = -1
        for feature in insts:
            curr_max_char_seq_len = max(feature.char_seq_lens)
            max_char_seq_len = max(curr_max_char_seq_len, max_char_seq_len)
        for i, feature in enumerate(insts):
            padding_length = max_seq_len - len(feature.words)
            words = feature.words + [0] * padding_length
            chars = []
            char_seq_lens = feature.char_seq_lens + [1] * padding_length
            for word_idx in range(feature.word_seq_len):
                pad_char_length = max_char_seq_len - feature.char_seq_lens[word_idx]
                word_chars = feature.chars[word_idx] + [0] * pad_char_length
                chars.append(word_chars)
            for _ in range(max_seq_len - feature.word_seq_len):
                chars.append([0] * max_char_seq_len)
            labels = (
                feature.labels + [0] * padding_length
                if feature.labels is not None
                else None
            )

            insts[i] = Feature(
                words=np.asarray(words),
                chars=np.asarray(chars),
                char_seq_lens=np.asarray(char_seq_lens),
                context_emb=feature.context_emb,
                word_seq_len=feature.word_seq_len,
                labels=np.asarray(labels) if labels is not None else None,
            )
        results = Feature(
            *(
                default_collate(samples) if not check_all_obj_is_None(samples) else None
                for samples in zip(*insts)
            )
        )
        return results

    def predict(self, sents: Union[str, List[str]]):
        sentences = [sents] if isinstance(sents, str) else sents
        insts = self.sents_to_insts(sentences)
        if self.conf.static_context_emb != ContextEmb.none:
            parse_elmo_vector(self.elmo, insts)
        test_batches = self.create_batch_data(insts)
        predictions = self.predict_insts(test_batches)
        if len(predictions) == 1:
            return predictions[0]
        else:
            return predictions

    def predict_and_display(self, sents: Union[str, List[str]]):
        sentences = [sents] if isinstance(sents, str) else sents
        predictions = self.predict(sentences)

        if isinstance(predictions[0], str):
            predictions = [predictions]

        insts = self.sents_to_insts(sentences)

        prefix_map = {"B-": "[BEGIN] ", "I-": "[INSIDE]", "E-": "[END]   "}

        for i, (inst, pred) in enumerate(zip(insts, predictions)):
            print(f"Sentence {i + 1}: {sentences[i]}")
            print("-" * 75)

            words = inst.words

            if len(words) != len(pred):
                print(
                    f"Warning: Mismatch between words ({len(words)}) and predictions ({len(pred)})"
                )
                min_len = min(len(words), len(pred))
                words = words[:min_len]
                pred = pred[:min_len]

            for j, (word, label) in enumerate(zip(words, pred)):
                label_prefix = "[OTHER] "
                entity_type = ""

                for prefix, prefix_label in prefix_map.items():
                    if label.startswith(prefix):
                        label_prefix = prefix_label
                        entity_type = label[len(prefix) :]
                        break

                display_label = (
                    f"{label_prefix} {entity_type}"
                    if entity_type
                    else f"{label_prefix} {label}"
                )
                print(f"  {j + 1:2d}. {word:20} -> {display_label}")

        return predictions if len(predictions) > 1 else predictions[0]

    def predict_and_display_with_entities(self, sents: Union[str, List[str]]):
        sentences = [sents] if isinstance(sents, str) else sents
        predictions = self.predict(sentences)

        if isinstance(predictions[0], str):
            predictions = [predictions]

        insts = self.sents_to_insts(sentences)
        all_entities = []

        prefix_map = {"B-": "[BEGIN] ", "I-": "[INSIDE]", "E-": "[END]   "}

        for i, (inst, pred) in enumerate(zip(insts, predictions)):
            print(f"Sentence {i + 1}: {sentences[i]}")
            print("-" * 75)

            words = inst.words

            if len(words) != len(pred):
                print(
                    f"Warning: Mismatch between words ({len(words)}) and predictions ({len(pred)})"
                )
                min_len = min(len(words), len(pred))
                words = words[:min_len]
                pred = pred[:min_len]

            for j, (word, label) in enumerate(zip(words, pred)):
                label_prefix = "[OTHER] "
                entity_type = ""

                for prefix, prefix_label in prefix_map.items():
                    if label.startswith(prefix):
                        label_prefix = prefix_label
                        entity_type = label[len(prefix) :]
                        break

                display_label = (
                    f"{label_prefix} {entity_type}"
                    if entity_type
                    else f"{label_prefix} {label}"
                )
                print(f"  {j + 1:2d}. {word:20} -> {display_label}")

            entities = self._extract_entities(words, pred)
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

        return {
            "predictions": predictions if len(predictions) > 1 else predictions[0],
            "entities": all_entities if len(all_entities) > 1 else all_entities[0],
        }

    def _extract_entities(
        self, words: List[str], labels: List[str]
    ) -> Dict[str, List[tuple]]:
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


def parse_elmo_vector(
    elmo, insts: List[Instance], mode: str = "average", batch_size=0
) -> None:
    """
    Attach the instances into the sentence/
    :param elmo: ELMo embedder
    :param insts: List of instance
    :param mode: the mode of elmo vectors
    :return:
    """
    all_sents = []
    for inst in insts:
        all_sents.append(inst.ori_words)
    if batch_size < 1:  # Not using batch
        for i, sent in tqdm(enumerate(all_sents), desc="Elmo Embedding"):
            elmo_vecs = elmo.embed_sentence(sent)
            vec = parse_sentence(elmo_vecs, mode=mode)
            insts[i].elmo_vec = vec
    else:  # Batched prediction
        for i, elmo_vecs in tqdm(
            enumerate(elmo.embed_sentences(all_sents, batch_size=batch_size)),
            desc="Elmo Embedding",
            total=len(all_sents),
        ):
            vec = parse_sentence(elmo_vecs, mode=mode)
            insts[i].elmo_vec = vec


if __name__ == "__main__":
    predictor = NERPredictor("model_files/english_model.tar.gz")
    res = predictor.predict("This is a demo")
    print(res)
