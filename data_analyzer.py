import os
import re
import xml.etree.ElementTree as ET

from collections import Counter

import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class DataAnalyzer:
    def __init__(self, xml_file_path, output_dir):
        self.xml_file_path = xml_file_path
        self.output_dir = output_dir
        self.tree = None
        self.root = None
        self.sentences = []
        self.entities = []
        self.tokens = []
        self.bio_tags = []
        self.load_and_parse()

    def load_and_parse(self):
        try:
            self.tree = ET.parse(self.xml_file_path)
            self.root = self.tree.getroot()
            print("Loaded XML file successfully")
            self._parse_ner_data()
        except Exception as e:
            print(f"Error loading XML: {e}")

    def _parse_ner_data(self):
        """Parse XML into NER-specific format with BIO tagging"""
        for stc in self.root.findall(".//STC"):
            stc_id = stc.get("ID")

            # Extract tokens and BIO tags
            tokens, bio_tags, entities = self._extract_bio_sequence(stc)

            if tokens:
                self.sentences.append(
                    {
                        "stc_id": stc_id,
                        "tokens": tokens,
                        "bio_tags": bio_tags,
                        "token_count": len(tokens),
                        "entity_count": len(
                            [tag for tag in bio_tags if tag.startswith("B-")]
                        ),
                        "entities": entities,
                    }
                )

                # Store token-level data
                for i, (token, tag) in enumerate(zip(tokens, bio_tags)):
                    self.tokens.append(
                        {
                            "stc_id": stc_id,
                            "position": i,
                            "token": token,
                            "bio_tag": tag,
                            "token_length": len(token),
                            "is_entity": tag != "O",
                        }
                    )

                # Store entity-level data
                for entity in entities:
                    self.entities.append(
                        {
                            "stc_id": stc_id,
                            "entity_type": entity["type"],
                            "entity_text": entity["text"],
                            "entity_length": len(entity["text"]),
                            "token_count": len(entity["text"].split()),
                            "start_pos": entity["start"],
                            "end_pos": entity["end"],
                        }
                    )

    def _extract_bio_sequence(self, stc_elem):
        """Extract BIO sequence from XML element"""
        tokens = []
        bio_tags = []
        entities = []
        current_pos = 0

        def process_element(elem, parent_entity_type=None):
            nonlocal current_pos

            # Process text before children
            if elem.text:
                words = elem.text.strip().split()
                for word in words:
                    if word:
                        tokens.append(word)
                        if parent_entity_type:
                            if (
                                len(
                                    [
                                        t
                                        for t in bio_tags
                                        if t.startswith(f"B-{parent_entity_type}")
                                        or t.startswith(f"I-{parent_entity_type}")
                                    ]
                                )
                                == 0
                            ):
                                bio_tags.append(f"B-{parent_entity_type}")
                            else:
                                bio_tags.append(f"I-{parent_entity_type}")
                        else:
                            bio_tags.append("O")
                        current_pos += 1

            # Process children
            for child in elem:
                if child.tag in ["LOC", "TME", "PER", "NUM", "TITLE"]:
                    # This is an entity
                    entity_start = len(tokens)
                    if child.text:
                        entity_words = child.text.strip().split()
                        entity_text = child.text.strip()

                        for i, word in enumerate(entity_words):
                            if word:
                                tokens.append(word)
                                if i == 0:
                                    bio_tags.append(f"B-{child.tag}")
                                else:
                                    bio_tags.append(f"I-{child.tag}")
                                current_pos += 1

                        entities.append(
                            {
                                "type": child.tag,
                                "text": entity_text,
                                "start": entity_start,
                                "end": len(tokens) - 1,
                            }
                        )
                else:
                    # Recursive processing for nested elements
                    process_element(child, parent_entity_type)

                # Process tail text
                if child.tail:
                    words = child.tail.strip().split()
                    for word in words:
                        if word:
                            tokens.append(word)
                            if parent_entity_type:
                                bio_tags.append(f"I-{parent_entity_type}")
                            else:
                                bio_tags.append("O")
                            current_pos += 1

        process_element(stc_elem)
        return tokens, bio_tags, entities

    def analyze_dataset_balance(self):
        """Analyze class balance - critical for NER"""
        print("PHÂN TÍCH CÂN BẰNG DỮ LIỆU")
        print("=" * 70)

        # BIO tag distribution
        bio_counter = Counter([token["bio_tag"] for token in self.tokens])
        total_tokens = len(self.tokens)

        print("Phân phối BIO tags:")
        for tag, count in bio_counter.most_common():
            percentage = (count / total_tokens) * 100
            print(f"   {tag:12}: {count:6} tokens ({percentage:5.2f}%)")

        # Entity type distribution
        entity_counter = Counter([ent["entity_type"] for ent in self.entities])
        total_entities = len(self.entities)

        print(f"\nPhân phối loại entities ({total_entities} entities):")
        for ent_type, count in entity_counter.most_common():
            percentage = (count / total_entities) * 100
            print(f"   {ent_type:12}: {count:6} entities ({percentage:5.2f}%)")

        # Calculate imbalance ratio
        non_entity_ratio = bio_counter["O"] / total_tokens
        entity_ratio = 1 - non_entity_ratio

        print("\nTỷ lệ cân bằng:")
        print(f"   - Non-entity (O): {non_entity_ratio:.3f}")
        print(f"   - Entity tokens: {entity_ratio:.3f}")
        print(f"   - Imbalance ratio: {non_entity_ratio / entity_ratio:.2f}:1")

        return bio_counter, entity_counter

    def analyze_entity_patterns(self):
        """Analyze entity patterns and characteristics"""
        print("\nPHÂN TÍCH ENTITY PATTERNS")
        print("=" * 70)

        # Entity length analysis
        entity_lengths = [ent["token_count"] for ent in self.entities]

        print("Phân tích độ dài entities:")
        print(f"   - Trung bình: {np.mean(entity_lengths):.2f} tokens")
        print(f"   - Median: {np.median(entity_lengths):.2f} tokens")
        print(f"   - Min: {min(entity_lengths)} tokens")
        print(f"   - Max: {max(entity_lengths)} tokens")
        print(f"   - Std: {np.std(entity_lengths):.2f}")

        # Length distribution by entity type
        print("\nĐộ dài trung bình theo loại entity:")
        for ent_type in set(ent["entity_type"] for ent in self.entities):
            type_lengths = [
                ent["token_count"]
                for ent in self.entities
                if ent["entity_type"] == ent_type
            ]
            print(
                f"   {ent_type:12}: {np.mean(type_lengths):.2f} ± {np.std(type_lengths):.2f} tokens"
            )

        # Multi-token entities analysis
        multi_token_entities = [ent for ent in self.entities if ent["token_count"] > 1]
        single_token_entities = [
            ent for ent in self.entities if ent["token_count"] == 1
        ]

        print("\nPhân tích entities đa token:")
        print(
            f"   - Single-token entities: {len(single_token_entities)} ({len(single_token_entities) / len(self.entities) * 100:.1f}%)"
        )
        print(
            f"   - Multi-token entities: {len(multi_token_entities)} ({len(multi_token_entities) / len(self.entities) * 100:.1f}%)"
        )

        # Most common entities
        entity_texts = Counter([ent["entity_text"] for ent in self.entities])
        print("\nTop 10 entities xuất hiện nhiều nhất:")
        for entity, count in entity_texts.most_common(10):
            print(f"   '{entity}': {count} lần")

    def analyze_sequence_patterns(self):
        """Analyze sequence patterns important for NER"""
        print("\nPHÂN TÍCH SEQUENCE PATTERNS")
        print("=" * 70)

        # Sentence length distribution
        sent_lengths = [sent["token_count"] for sent in self.sentences]

        print("Phân tích độ dài câu:")
        print(f"   - Trung bình: {np.mean(sent_lengths):.2f} tokens")
        print(f"   - Median: {np.median(sent_lengths):.2f} tokens")
        print(f"   - Min: {min(sent_lengths)} tokens")
        print(f"   - Max: {max(sent_lengths)} tokens")
        print(f"   - Percentile 95%: {np.percentile(sent_lengths, 95):.0f} tokens")

        # Entity density analysis
        entity_densities = [
            sent["entity_count"] / sent["token_count"] for sent in self.sentences
        ]

        print("\nMật độ entities trong câu:")
        print(f"   - Trung bình: {np.mean(entity_densities):.3f}")
        print(f"   - Median: {np.median(entity_densities):.3f}")
        print(f"   - Max: {max(entity_densities):.3f}")

        # Adjacent entity patterns
        adjacent_patterns = self._find_adjacent_entity_patterns()
        print("\nPatterns entities liền kề:")
        for pattern, count in adjacent_patterns.most_common(5):
            print(f"   {pattern}: {count} lần")

    def _find_adjacent_entity_patterns(self):
        """Find patterns of adjacent entities"""
        patterns = []

        for sent in self.sentences:
            bio_tags = sent["bio_tags"]
            current_entities = []

            for tag in bio_tags:
                if tag.startswith("B-"):
                    if current_entities:
                        if len(current_entities) > 1:
                            patterns.append(" -> ".join(current_entities))
                        current_entities = []
                    current_entities.append(tag[2:])
                elif tag.startswith("I-"):
                    continue
                else:  # 'O'
                    if current_entities and len(current_entities) > 1:
                        patterns.append(" -> ".join(current_entities))
                    current_entities = []

            if current_entities and len(current_entities) > 1:
                patterns.append(" -> ".join(current_entities))

        return Counter(patterns)

    def analyze_vocabulary_characteristics(self):
        """Analyze vocabulary characteristics important for NER"""
        print("\nPHÂN TÍCH TỪ VỰNG")
        print("=" * 70)

        # Overall vocabulary
        all_tokens = [token["token"] for token in self.tokens]
        vocab = set(all_tokens)
        token_counter = Counter(all_tokens)

        print("Thống kê từ vựng:")
        print(f"   - Tổng tokens: {len(all_tokens):,}")
        print(f"   - Unique tokens: {len(vocab):,}")
        print(f"   - Vocabulary richness: {len(vocab) / len(all_tokens):.4f}")

        # Entity vs non-entity vocabulary
        entity_tokens = [token["token"] for token in self.tokens if token["is_entity"]]
        non_entity_tokens = [
            token["token"] for token in self.tokens if not token["is_entity"]
        ]

        entity_vocab = set(entity_tokens)
        non_entity_vocab = set(non_entity_tokens)

        print("\nTừ vựng Entity vs Non-entity:")
        print(f"   - Entity tokens: {len(entity_tokens):,}")
        print(f"   - Entity vocabulary: {len(entity_vocab):,}")
        print(f"   - Non-entity tokens: {len(non_entity_tokens):,}")
        print(f"   - Non-entity vocabulary: {len(non_entity_vocab):,}")
        print(f"   - Vocabulary overlap: {len(entity_vocab & non_entity_vocab):,}")

        # Out-of-vocabulary simulation
        self._analyze_oov_potential(token_counter)

        # Special characters and patterns
        self._analyze_special_patterns()

    def _analyze_oov_potential(self, token_counter):
        """Analyze potential OOV issues"""
        print("\nPhân tích khả năng OOV:")

        # Hapax legomena (words appearing only once)
        hapax = [token for token, count in token_counter.items() if count == 1]
        print(
            f"   - Hapax legomena: {len(hapax)} ({len(hapax) / len(token_counter) * 100:.2f}%)"
        )

        # Low frequency words (appearing <= 2 times)
        low_freq = [token for token, count in token_counter.items() if count <= 2]
        print(
            f"   - Low frequency (≤2): {len(low_freq)} ({len(low_freq) / len(token_counter) * 100:.2f}%)"
        )

        # Most common tokens
        print("\nTop 10 tokens phổ biến nhất:")
        for token, count in token_counter.most_common(10):
            print(f"   '{token}': {count} lần")

    def _analyze_special_patterns(self):
        """Analyze special character patterns"""
        all_tokens = [token["token"] for token in self.tokens]

        # Pattern analysis
        numeric_tokens = [t for t in all_tokens if re.search(r"\d", t)]
        punctuated_tokens = [t for t in all_tokens if re.search(r"[^\w\s]", t)]
        capitalized_tokens = [t for t in all_tokens if t[0].isupper()]

        print("\nPatterns đặc biệt:")
        print(
            f"   - Chứa số: {len(numeric_tokens)} ({len(numeric_tokens) / len(all_tokens) * 100:.2f}%)"
        )
        print(
            f"   - Chứa dấu câu: {len(punctuated_tokens)} ({len(punctuated_tokens) / len(all_tokens) * 100:.2f}%)"
        )
        print(
            f"   - Viết hoa đầu: {len(capitalized_tokens)} ({len(capitalized_tokens) / len(all_tokens) * 100:.2f}%)"
        )

    def visualize_ner_characteristics(self):
        """Create NER-specific visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("Phân Tích Đặc Trưng Dữ Liệu NER", fontsize=16, fontweight="bold")

        # 1. BIO tag distribution
        bio_tags = [token["bio_tag"] for token in self.tokens]
        bio_counter = Counter(bio_tags)

        axes[0, 0].bar(bio_counter.keys(), bio_counter.values(), color="skyblue")
        axes[0, 0].set_title("Phân phối BIO Tags")
        axes[0, 0].set_ylabel("Số lượng")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Entity type distribution
        entity_types = [ent["entity_type"] for ent in self.entities]
        entity_counter = Counter(entity_types)

        colors = plt.cm.Set3(np.linspace(0, 1, len(entity_counter)))
        axes[0, 1].pie(
            entity_counter.values(),
            labels=entity_counter.keys(),
            autopct="%1.1f%%",
            colors=colors,
        )
        axes[0, 1].set_title("Phân phối loại Entities")

        # 3. Sentence length distribution
        sent_lengths = [sent["token_count"] for sent in self.sentences]
        axes[0, 2].hist(
            sent_lengths, bins=20, alpha=0.7, color="lightgreen", edgecolor="black"
        )
        axes[0, 2].set_title("Phân phối độ dài câu")
        axes[0, 2].set_xlabel("Số tokens")
        axes[0, 2].set_ylabel("Tần suất")
        axes[0, 2].axvline(
            np.mean(sent_lengths),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(sent_lengths):.1f}",
        )
        axes[0, 2].legend()

        # 4. Entity length distribution by type
        entity_types_unique = list(set(ent["entity_type"] for ent in self.entities))
        for i, ent_type in enumerate(entity_types_unique):
            lengths = [
                ent["token_count"]
                for ent in self.entities
                if ent["entity_type"] == ent_type
            ]
            axes[1, 0].hist(
                lengths, alpha=0.6, label=ent_type, bins=range(1, max(lengths) + 2)
            )

        axes[1, 0].set_title("Phân phối độ dài Entity theo loại")
        axes[1, 0].set_xlabel("Số tokens")
        axes[1, 0].set_ylabel("Tần suất")
        axes[1, 0].legend()

        # 5. Entity density per sentence
        entity_densities = [
            sent["entity_count"] / sent["token_count"] for sent in self.sentences
        ]
        axes[1, 1].hist(
            entity_densities, bins=20, alpha=0.7, color="orange", edgecolor="black"
        )
        axes[1, 1].set_title("Mật độ Entity trong câu")
        axes[1, 1].set_xlabel("Tỷ lệ entity/token")
        axes[1, 1].set_ylabel("Tần suất")

        # 6. Token length distribution (character level)
        token_lengths = [token["token_length"] for token in self.tokens]
        axes[1, 2].hist(
            token_lengths, bins=20, alpha=0.7, color="purple", edgecolor="black"
        )
        axes[1, 2].set_title("Phân phối độ dài Token")
        axes[1, 2].set_xlabel("Số ký tự")
        axes[1, 2].set_ylabel("Tần suất")

        plt.tight_layout()
        plt.show()

    def generate_word_clouds(self):
        """Generates word clouds: one for all tokens and one for entity tokens."""
        all_tokens = [token["token"] for token in self.tokens]
        entity_tokens = [token["token"] for token in self.tokens if token["is_entity"]]

        if not all_tokens and not entity_tokens:
            print("Không có tokens nào để tạo word clouds.")
            return

        plt.figure(figsize=(15, 7))

        # Word Cloud for All Tokens
        if all_tokens:
            text_all = " ".join(all_tokens)
            wordcloud_all = WordCloud(
                width=600, height=300, background_color="white"
            ).generate(text_all)
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(wordcloud_all, interpolation="bilinear")
            ax1.set_title("Word Cloud của tất cả token")
            ax1.axis("off")
        else:
            ax1 = plt.subplot(1, 2, 1)
            ax1.set_title("Word Cloud của tất cả token")
            ax1.text(
                0.5,
                0.5,
                "No tokens to display",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax1.transAxes,
            )
            ax1.axis("off")

        # Word Cloud for Entity Tokens
        if entity_tokens:
            text_entities = " ".join(entity_tokens)
            wordcloud_entities = WordCloud(
                width=600, height=300, background_color="white"
            ).generate(text_entities)
            ax2 = plt.subplot(1, 2, 2)
            ax2.imshow(wordcloud_entities, interpolation="bilinear")
            ax2.set_title("Word Cloud của các token Entity")
            ax2.axis("off")
        else:
            ax2 = plt.subplot(1, 2, 2)
            ax2.set_title("Word Cloud của các token entity")
            ax2.text(
                0.5,
                0.5,
                "No entity tokens to display",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax2.transAxes,
            )
            ax2.axis("off")

        plt.tight_layout()
        plt.show()

    def analyze_model_readiness(self):
        """Analyze data readiness for NER models"""
        print("\nĐÁNH GIÁ ĐỘ SẴN SÀNG CHO MÔ HÌNH")
        print("=" * 70)

        # Data size assessment
        total_tokens = len(self.tokens)
        total_sentences = len(self.sentences)
        total_entities = len(self.entities)

        print("Đánh giá kích thước dữ liệu:")
        print(f"   - Tổng tokens: {total_tokens:,}")
        print(f"   - Tổng câu: {total_sentences:,}")
        print(f"   - Tổng entities: {total_entities:,}")

        # Class balance assessment
        bio_counter = Counter([token["bio_tag"] for token in self.tokens])
        o_ratio = bio_counter["O"] / total_tokens
        print(f"Tỷ lệ non-entity: {o_ratio * 100:.2f}%")

        # Entity coverage assessment
        entity_coverage = len(set(ent["entity_text"] for ent in self.entities))
        print("\nĐánh giá coverage entities:")
        print(f"   - Unique entities: {entity_coverage}")
        print(f"   - Entities/câu trung bình: {total_entities / total_sentences:.2f}")

        # Suggested train/dev/test split
        print("\nĐề xuất chia dữ liệu:")
        print(f"   - Train: {int(total_sentences * 0.7)} câu (70%)")
        print(f"   - Dev: {int(total_sentences * 0.1)} câu (10%)")
        print(f"   - Test: {int(total_sentences * 0.2)} câu (20%)")

    def export_ner_analysis(self):
        """Export analysis results for further processing"""
        # Create comprehensive analysis report
        analysis_data = {
            "dataset_stats": {
                "total_sentences": len(self.sentences),
                "total_tokens": len(self.tokens),
                "total_entities": len(self.entities),
                "avg_sentence_length": np.mean(
                    [s["token_count"] for s in self.sentences]
                ),
                "avg_entities_per_sentence": np.mean(
                    [s["entity_count"] for s in self.sentences]
                ),
            },
            "bio_distribution": Counter([token["bio_tag"] for token in self.tokens]),
            "entity_distribution": Counter(
                [ent["entity_type"] for ent in self.entities]
            ),
            "vocabulary_size": len(set(token["token"] for token in self.tokens)),
        }
        print(analysis_data)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Export detailed data
        pd.DataFrame(self.sentences).to_csv(
            os.path.join(self.output_dir, "ner_sentences.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(self.tokens).to_csv(
            os.path.join(self.output_dir, "ner_tokens.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(self.entities).to_csv(
            os.path.join(self.output_dir, "ner_entities.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        print("\nĐã export các file phân tích:")
        print("   - ner_sentences.csv: Dữ liệu câu")
        print("   - ner_tokens.csv: Dữ liệu token-level")
        print("   - ner_entities.csv: Dữ liệu entity-level")

    def run_complete_ner_analysis(self):
        print("BẮT ĐẦU PHÂN TÍCH DỮ LIỆU")
        print("=" * 70)

        # Core NER analyses
        self.analyze_dataset_balance()
        self.analyze_entity_patterns()
        self.analyze_sequence_patterns()
        self.analyze_vocabulary_characteristics()
        self.analyze_model_readiness()

        # Visualizations
        print("\nTạo visualizations...")
        self.visualize_ner_characteristics()
        self.generate_word_clouds()

        # Export results
        self.export_ner_analysis()
