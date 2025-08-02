from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from src.data.data_utils import convert_iobes, build_label_idx

from src.data import Instance
import logging
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.getLogger(__name__)


def convert_instances_to_feature_tensors(
    instances: List[Instance],
    tokenizer,
    label2idx: Dict[str, int],
    max_length: int = 256,
) -> List[Dict]:
    features = []
    ## tokenize the word into word_piece / BPE
    ## NOTE: adding a leading space is important for BART/GPT/Roberta tokenization.
    ## Related GitHub issues:
    ##      https://github.com/huggingface/transformers/issues/1196
    ##      https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py#L38-L56
    ##      https://github.com/ThilinaRajapakse/simpletransformers/issues/458
    # assert tokenizer.add_prefix_space ## has to be true, in order to tokenize pre-tokenized input
    logger.info(f"[Data Info] Using max_length={max_length} for tokenization")

    # Kiểm tra xem tokenizer có phải là fast tokenizer không
    is_fast_tokenizer = isinstance(tokenizer, PreTrainedTokenizerFast)

    for idx, inst in enumerate(instances):
        words = inst.ori_words
        orig_to_tok_index = []

        if is_fast_tokenizer:
            # Sử dụng fast tokenizer với max_length
            res = tokenizer.encode_plus(
                words,
                is_split_into_words=True,
                max_length=max_length,
                padding=False,
                truncation=True,
                return_overflowing_tokens=False,
            )

            subword_idx2word_idx = res.word_ids(batch_index=0)
            prev_word_idx = -1
            word_count = 0

            for i, mapped_word_idx in enumerate(subword_idx2word_idx):
                """
                Note: by default, we use the first wordpiece/subword token to represent the word
                If you want to do something else (e.g., use last wordpiece to represent), modify them here.
                """
                if mapped_word_idx is None:  ## cls and sep token
                    continue
                if mapped_word_idx != prev_word_idx:
                    ## because we take the first subword to represent the whole word
                    if word_count < len(words):  # Đảm bảo không vượt quá số từ gốc
                        orig_to_tok_index.append(i)
                        word_count += 1
                    prev_word_idx = mapped_word_idx

            # Truncate labels if necessary
            if len(orig_to_tok_index) > len(words):
                orig_to_tok_index = orig_to_tok_index[: len(words)]

        else:
            # Sử dụng slow tokenizer - cần xử lý thủ công
            all_tokens = []
            truncated_words = []

            for word_idx, word in enumerate(words):
                # Tokenize từng từ riêng lẻ
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) == 0:
                    # Nếu từ không thể tokenize, thêm [UNK]
                    word_tokens = [tokenizer.unk_token]

                # Kiểm tra xem có vượt quá max_length không (tính cả [CLS] và [SEP])
                if len(all_tokens) + len(word_tokens) + 2 > max_length:
                    break

                # Lưu index của token đầu tiên của từ này
                orig_to_tok_index.append(len(all_tokens) + 1)  # +1 để tính [CLS] token
                all_tokens.extend(word_tokens)
                truncated_words.append(word)

            # Tạo input_ids với [CLS] và [SEP]
            input_ids = [tokenizer.cls_token_id]
            input_ids.extend(tokenizer.convert_tokens_to_ids(all_tokens))
            input_ids.append(tokenizer.sep_token_id)

            # Tạo attention_mask
            attention_mask = [1] * len(input_ids)

            res = {"input_ids": input_ids, "attention_mask": attention_mask}
            words = truncated_words  # Cập nhật words list nếu bị truncate

        # Đảm bảo orig_to_tok_index và words có cùng độ dài
        if len(orig_to_tok_index) != len(words):
            min_len = min(len(orig_to_tok_index), len(words))
            orig_to_tok_index = orig_to_tok_index[:min_len]
            words = words[:min_len]

        assert len(orig_to_tok_index) == len(words), (
            f"Length mismatch: orig_to_tok_index={len(orig_to_tok_index)}, words={len(words)}"
        )

        labels = inst.labels
        if labels:
            # Truncate labels to match words length
            labels = labels[: len(words)]
            label_ids = [label2idx[label] for label in labels]
        else:
            label_ids = [-100] * len(words)

        segment_ids = [0] * len(res["input_ids"])

        features.append(
            {
                "input_ids": res["input_ids"],
                "attention_mask": res["attention_mask"],
                "orig_to_tok_index": orig_to_tok_index,
                "token_type_ids": segment_ids,
                "word_seq_len": len(orig_to_tok_index),
                "label_ids": label_ids,
            }
        )
    return features


class TransformersNERDataset(Dataset):
    def __init__(
        self,
        file: str,
        tokenizer,
        is_train: bool,
        sents: List[List[str]] = None,
        label2idx: Dict[str, int] = None,
        number: int = -1,
        max_length: int = 256,
    ):
        """
        sents: we use sentences if we want to build dataset from sentences directly instead of file
        max_length: maximum sequence length for tokenization
        """
        ## read all the instances. sentences and labels
        insts = (
            self.read_file(file=file, number=number)
            if sents is None
            else self.read_from_sentences(sents)
        )
        self.insts = insts
        self.max_length = max_length

        if is_train:
            # assert label2idx is None
            if label2idx is not None:
                logger.warning(
                    "YOU ARE USING EXTERNAL label2idx, WHICH IS NOT BUILT FROM TRAINING SET."
                )
                self.label2idx = label2idx
            else:
                logger.info("[Data Info] Using the training set to build label index")
                ## build label to index mapping. e.g., B-PER -> 0, I-PER -> 1
                idx2labels, label2idx = build_label_idx(insts)
                self.idx2labels = idx2labels
                self.label2idx = label2idx
        else:
            assert (
                label2idx is not None
            )  ## for dev/test dataset we don't build label2idx
            self.label2idx = label2idx
            # check_all_labels_in_dict(insts=insts, label2idx=self.label2idx)

        self.insts_ids = convert_instances_to_feature_tensors(
            insts, tokenizer, label2idx, max_length
        )
        self.tokenizer = tokenizer

    def read_from_sentences(self, sents: List[List[str]]):
        """
        sents = [['word_a', 'word_b'], ['word_aaa', 'word_bccc', 'word_ccc']]
        """
        insts = []
        for sent in sents:
            insts.append(Instance(words=sent, ori_words=sent))
        return insts

    def read_file(self, file: str, number: int = -1) -> List[Instance]:
        logger.info(
            f"[Data Info] Reading file: {file}, labels will be converted to IOBES encoding"
        )
        logger.info(
            "[Data Info] Modify src/data/transformers_dataset.read_txt function if you have other requirements"
        )
        insts = []
        with open(file, "r", encoding="utf-8") as f:
            words = []
            ori_words = []
            labels = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    if words:
                        labels = convert_iobes(labels)
                        insts.append(
                            Instance(words=words, ori_words=ori_words, labels=labels)
                        )
                    words = []
                    ori_words = []
                    labels = []
                    if len(insts) == number:
                        break
                    continue
                ls = line.split()
                if len(ls) >= 2:
                    word, label = ls[0], ls[-1]
                    ori_words.append(word)
                    words.append(word)
                    labels.append(label)

            # Xử lý instance cuối cùng nếu file không kết thúc bằng dòng trống
            if words:
                labels = convert_iobes(labels)
                insts.append(Instance(words=words, ori_words=ori_words, labels=labels))

        logger.info(f"number of sentences: {len(insts)}")
        return insts

    def __len__(self):
        return len(self.insts_ids)

    def __getitem__(self, index):
        return self.insts_ids[index]

    def collate_fn(self, batch: List[Dict]):
        # Kiểm tra tính hợp lệ của batch
        valid_batch = []
        for feature in batch:
            if (
                len(feature["orig_to_tok_index"])
                == len(feature["label_ids"])
                == feature["word_seq_len"]
            ):
                valid_batch.append(feature)
            else:
                logger.warning(
                    f"Skipping invalid feature: orig_to_tok_index={len(feature['orig_to_tok_index'])}, "
                    f"label_ids={len(feature['label_ids'])}, word_seq_len={feature['word_seq_len']}"
                )

        if not valid_batch:
            raise ValueError("No valid features in batch")

        batch = valid_batch

        word_seq_len = [len(feature["orig_to_tok_index"]) for feature in batch]
        max_seq_len = max(word_seq_len)
        max_wordpiece_length = max([len(feature["input_ids"]) for feature in batch])

        # Giới hạn max_wordpiece_length để tránh lỗi memory
        if max_wordpiece_length > self.max_length:
            max_wordpiece_length = self.max_length

        for i, feature in enumerate(batch):
            # Truncate nếu cần thiết
            if len(feature["input_ids"]) > max_wordpiece_length:
                feature["input_ids"] = feature["input_ids"][:max_wordpiece_length]
                feature["attention_mask"] = feature["attention_mask"][
                    :max_wordpiece_length
                ]
                feature["token_type_ids"] = feature["token_type_ids"][
                    :max_wordpiece_length
                ]

            padding_length = max_wordpiece_length - len(feature["input_ids"])
            input_ids = (
                feature["input_ids"] + [self.tokenizer.pad_token_id] * padding_length
            )
            mask = feature["attention_mask"] + [0] * padding_length
            type_ids = (
                feature["token_type_ids"]
                + [self.tokenizer.pad_token_type_id] * padding_length
            )
            padding_word_len = max_seq_len - len(feature["orig_to_tok_index"])
            orig_to_tok_index = feature["orig_to_tok_index"] + [0] * padding_word_len
            label_ids = feature["label_ids"] + [0] * padding_word_len

            batch[i] = {
                "input_ids": input_ids,
                "attention_mask": mask,
                "token_type_ids": type_ids,
                "orig_to_tok_index": orig_to_tok_index,
                "word_seq_len": feature["word_seq_len"],
                "label_ids": label_ids,
            }

        encoded_inputs = {
            key: [example[key] for example in batch] for key in batch[0].keys()
        }

        # Convert to tensors với error handling
        try:
            results = BatchEncoding(encoded_inputs, tensor_type="pt")
        except Exception as e:
            logger.error(f"Error creating batch encoding: {e}")
            # Debug info
            for key, values in encoded_inputs.items():
                lengths = [len(v) if isinstance(v, list) else v for v in values]
                logger.error(f"{key}: {lengths}")
            raise

        return results


## testing code to test the dataset
if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    dataset = TransformersNERDataset(
        file="data/vietnam-history-data-ner/dev.txt",
        tokenizer=tokenizer,
        is_train=True,
        max_length=256,
    )
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        num_workers=2,
        collate_fn=dataset.collate_fn,
    )
    print(len(train_dataloader))
    for batch in train_dataloader:
        # print(batch.input_ids.size())
        print(batch.input_ids)
        break
