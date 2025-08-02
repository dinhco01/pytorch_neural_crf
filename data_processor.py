import os

from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, data_dir: str = "output_data", seed: int = 42):
        self.data_dir = data_dir
        self.seed = seed

        # Data containers
        self.sentences_df = None

        # Split data
        self.train_sentences = None
        self.dev_sentences = None
        self.test_sentences = None

    def load_data(self):
        """Load the exported CSV files"""
        try:
            self.sentences_df = pd.read_csv(
                os.path.join(self.data_dir, "ner_sentences.csv"), encoding="utf-8-sig"
            )

            print("Đã load thành công file dữ liệu:")
            print(f"   - Sentences: {len(self.sentences_df)} câu")

            # Parse string representations back to lists
            self._parse_list_columns()

        except Exception as e:
            print(f"Lỗi khi load dữ liệu: {e}")
            raise

    def _parse_list_columns(self):
        """Parse string representations of lists back to actual lists"""
        import ast

        def safe_literal_eval(val):
            if pd.isna(val):
                return []
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return []

        # Parse tokens and bio_tags columns
        self.sentences_df["tokens"] = self.sentences_df["tokens"].apply(
            safe_literal_eval
        )
        self.sentences_df["bio_tags"] = self.sentences_df["bio_tags"].apply(
            safe_literal_eval
        )

        print("Đã parse các cột dạng list")

    def create_stratified_split(
        self, train_ratio: float = 0.7, dev_ratio: float = 0.1, test_ratio: float = 0.2
    ):
        """
        Chia dữ liệu với stratification dựa trên:
        - Số lượng entities trong câu
        - Loại entities chính trong câu
        """

        if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Tổng tỷ lệ phải bằng 1.0")

        print(f"\nChia dữ liệu theo tỷ lệ {train_ratio}:{dev_ratio}:{test_ratio}")

        # Tạo stratification key dựa trên đặc trưng của câu
        stratify_keys = []
        for _, row in self.sentences_df.iterrows():
            entity_count_bin = min(row["entity_count"], 5)  # Cap at 5+ entities

            # Lấy entity type phổ biến nhất trong câu
            bio_tags = row["bio_tags"]
            entity_types = [
                tag.split("-")[1] for tag in bio_tags if tag.startswith("B-")
            ]

            if entity_types:
                main_entity_type = Counter(entity_types).most_common(1)[0][0]
            else:
                main_entity_type = "NONE"

            stratify_key = f"{entity_count_bin}_{main_entity_type}"
            stratify_keys.append(stratify_key)

        self.sentences_df["stratify_key"] = stratify_keys

        # Thực hiện split 2 bước
        # Bước 1: Tách train vs (dev+test)
        train_sentences, temp_sentences = train_test_split(
            self.sentences_df,
            test_size=(dev_ratio + test_ratio),
            random_state=self.seed,
            stratify=self.sentences_df["stratify_key"],
        )

        # Bước 2: Tách dev vs test
        relative_test_size = test_ratio / (dev_ratio + test_ratio)
        dev_sentences, test_sentences = train_test_split(
            temp_sentences,
            test_size=relative_test_size,
            random_state=self.seed,
            stratify=temp_sentences["stratify_key"],
        )

        # Lưu kết quả
        self.train_sentences = train_sentences.drop("stratify_key", axis=1).reset_index(
            drop=True
        )
        self.dev_sentences = dev_sentences.drop("stratify_key", axis=1).reset_index(
            drop=True
        )
        self.test_sentences = test_sentences.drop("stratify_key", axis=1).reset_index(
            drop=True
        )

        print("Hoàn thành chia dữ liệu!")

        # In thống kê
        self._print_split_statistics()

    def _print_split_statistics(self):
        """In thống kê chi tiết của các split"""
        print("\nTHỐNG KÊ CÁC SPLIT")
        print("=" * 60)

        splits = [
            ("Train", self.train_sentences),
            ("Dev", self.dev_sentences),
            ("Test", self.test_sentences),
        ]

        for split_name, sentences_df in splits:
            total_tokens = sum(len(tokens) for tokens in sentences_df["tokens"])
            total_entities = sentences_df["entity_count"].sum()

            # Đếm distribution của entities
            entity_dist = Counter()
            for bio_tags in sentences_df["bio_tags"]:
                for tag in bio_tags:
                    if tag.startswith("B-"):
                        entity_type = tag.split("-")[1]
                        entity_dist[entity_type] += 1

            print(f"\n{split_name.upper()} SET:")
            print(f"   - Sentences: {len(sentences_df):,}")
            print(f"   - Tokens: {total_tokens:,}")
            print(f"   - Entities: {total_entities:,}")
            print(f"   - Avg sentence length: {sentences_df['token_count'].mean():.2f}")
            print(
                f"   - Avg entities/sentence: {sentences_df['entity_count'].mean():.2f}"
            )

            if entity_dist:
                print("   - Entity distribution:")
                for ent_type, count in entity_dist.most_common():
                    percentage = (
                        count / total_entities * 100 if total_entities > 0 else 0
                    )
                    print(f"     * {ent_type}: {count} ({percentage:.1f}%)")

    def _sentences_to_conll_format(self, sentences_df):
        """Chuyển đổi sentences DataFrame thành format CoNLL"""
        conll_lines = []

        for _, row in sentences_df.iterrows():
            tokens = row["tokens"]
            bio_tags = row["bio_tags"]

            # Thêm các token và tag của câu
            for token, tag in zip(tokens, bio_tags):
                conll_lines.append(f"{token}\t{tag}")

            # Thêm dòng trống để ngăn cách các câu
            conll_lines.append("")

        return conll_lines

    def save_conll_files(self, output_dir: str = "."):
        """Lưu các file CoNLL format"""
        os.makedirs(output_dir, exist_ok=True)

        # Tạo các file CoNLL
        splits = [
            ("train.txt", self.train_sentences),
            ("dev.txt", self.dev_sentences),
            ("test.txt", self.test_sentences),
        ]

        for filename, sentences_df in splits:
            conll_lines = self._sentences_to_conll_format(sentences_df)

            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(conll_lines))

            print(f"Đã lưu: {filepath} ({len(sentences_df)} sentences)")

        print(f"\nĐã lưu tất cả files CoNLL vào: {output_dir}/")

    def process_complete_pipeline(
        self,
        train_ratio: float = 0.7,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.2,
        output_dir: str = ".",
    ):
        print("BẮT ĐẦU PIPELINE XỬ LÝ DỮ LIỆU - CoNLL FORMAT")
        print("=" * 80)

        self.load_data()
        self.create_stratified_split(train_ratio, dev_ratio, test_ratio)
        self.save_conll_files(output_dir)
