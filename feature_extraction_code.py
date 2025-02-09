from ltp import LTP
import stanza
import torch
import numpy as np
import pandas as pd
import re
from collections import Counter

# Read the Excel file
data = pd.read_excel(f'.xlsx')

# Processing levels
lexical_level = True
sentence_level = True
textual_level = True

# Initialize LTP and Stanza
ltp = LTP()
nlp = stanza.Pipeline(lang='zh-hans', tokenize_pretokenized=True, download_method=None)

# Move the model to GPU if available
if torch.cuda.is_available():
    ltp.to("cuda")

# Function to load words from a TXT file
def load_words_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Remove newline characters and return as a list
        words = [line.strip() for line in file.readlines()]
    return words

# Load positive and negative words from text files
positive_words = load_words_from_file('正面词_简体.txt')
negative_words = load_words_from_file('负面词_简体.txt')
# Define first-person pronouns
first_person_words = ['我', '我们', '咱', '咱们', '俺', '俺们']
negation_words = ['不', '没', '无', '非', '否', '勿', '不是', '不会', '不能', '没有', '没能', '没法']
# Add custom words to the dictionary
ltp.add_words(["XX大学", "被试"], freq=2)

# Initialize a list to store features
features = []
participant_num = 0

# Function to clean sentences
def clean_sentences(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        # Remove invisible characters and trim whitespace
        cleaned_sentence = re.sub(r'[^\w一-龥，。？！、；：“”‘’（）《》【】]', '', sentence).strip()
        if cleaned_sentence:  # Ensure the sentence is not empty
            cleaned_sentences.append(cleaned_sentence)
    return cleaned_sentences

# Function to split text into segments
def split_text(text, max_length=300):
    segments = []
    start = 0
    while start < len(text):
        # Look for the last punctuation mark within the max_length range
        end = start + max_length
        if end < len(text):
            segment_end = max(text.rfind('。', start, end),
                              text.rfind('？', start, end),
                              text.rfind('！', start, end),
                              text.rfind('；', start, end),
                              text.rfind('：', start, end))
            if segment_end == -1:  # If no punctuation found, cut off directly
                segment_end = end
            else:
                segment_end += 1  # Include the punctuation mark
        else:
            segment_end = len(text)  # Last segment
        segments.append(text[start:segment_end])
        start = segment_end
    return segments


# Process data row by row
for index, row in data.iterrows():
    text = row['text']
    number = row['number']
    self_esteem = row['self-esteem']
    print(participant_num+1)

    # Split the text into segments
    text_segments = split_text(text, max_length=300)
    tokens, pos_tags, dep_relations = [], [], []

    # Process each text segment
    for segment in text_segments:
        try:
            # Perform tokenization, POS tagging, and dependency parsing with LTP
            pipeline_segment = ltp.pipeline([segment], tasks=['cws', 'pos', 'dep'])
            tokens.extend(pipeline_segment.cws[0])  # Merge tokenization results
            pos_tags.extend(pipeline_segment.pos[0])  # Merge POS tagging results
            dep_relations.extend(pipeline_segment.dep[0])   # Merge dependency parsing results
        except RuntimeError as e:
            print(f"Error processing segment: {segment}\nError: {e}")

    # First split the text into sentences, then clean them
    sentences = re.split(r'[。？！；：]', text)
    cleaned_sentences = clean_sentences(sentences)
    print(sentences)

    # Process dependency relations for each sentence
    segments = []
    dep_relations = []
    for sentence in cleaned_sentences:
        try:
            pipeline_sentence = ltp.pipeline([sentence], tasks=['cws', 'pos', 'dep'])
            segments.append(pipeline_sentence.cws[0])
            dep_relations.append(pipeline_sentence.dep[0])
        except RuntimeError as e:
            print(f"Error processing sentence: {sentence}\nError: {e}")

    # Initialize the feature dictionary
    feature_dict = {'number': number}

    if lexical_level:
        # Remove punctuation before counting words
        tokens_no_punctuation = [token for token in tokens if re.match(r'\w', token)]
        print(tokens_no_punctuation)

        # Use Counter to avoid redundant calculations
        word_counter = Counter(tokens_no_punctuation)

        # 1. Total word count
        total_word_count = len(tokens_no_punctuation)
        feature_dict['1_total_word_count'] = total_word_count

        # 2. First-person pronoun term frequency (TF)
        first_person_count = sum([word_counter[word] for word in first_person_words])
        first_person_tf = first_person_count / total_word_count
        feature_dict['2_first_person_tf'] = first_person_tf

        # 3. Negation term frequency (TF)
        negation_count = sum([word_counter[word] for word in negation_words])
        negation_tf = negation_count / total_word_count
        feature_dict['3_negation_tf'] = negation_tf

        # 4. Positive emotion word term frequency (TF)
        positive_count = sum([word_counter[word] for word in positive_words])
        positive_tf = positive_count / total_word_count
        feature_dict['4_positive_tf'] = positive_tf

        # 5. Negative emotion word term frequency (TF)
        negative_count = sum([word_counter[word] for word in negative_words])
        negative_tf = negative_count  / total_word_count
        feature_dict['5_negative_tf'] = negative_tf

    if sentence_level:
        # Remove punctuation from sentences
        sentences_no_punctuation_for_syntax = [re.sub(r'[^\w一-鿿]', '', ' '.join(segment)) for segment in segments if
                                           segment]
        segment_result = ltp.pipeline(sentences_no_punctuation_for_syntax, tasks=['cws'])
        SEG = segment_result.cws

        # 6. Total number of sentences
        total_sentences = len(sentences_no_punctuation_for_syntax)
        feature_dict['6_total_sentences'] = total_sentences

        # 7. Average sentence length
        sentence_lengths = [len(segment) for segment in SEG]
        mean_sentence_length = np.mean(sentence_lengths)
        feature_dict['7_mean_sentence_length'] = mean_sentence_length

        # 8. Average dependency tree depth
        def calc_tree_depth(dep_relation):
            heads = dep_relation['head']
            max_depth = 0
            for i in range(len(heads)):
                depth = 0
                current = i + 1
                visited = set()  # Prevent infinite loops
                while current != 0 and current <= len(heads) and current not in visited:
                    visited.add(current)
                    depth += 1
                    current = heads[current - 1]
                if depth > max_depth:
                    max_depth = depth
            return max_depth


        tree_depths = [calc_tree_depth(dep) for dep in dep_relations]
        mean_tree_depth = np.mean(tree_depths)
        feature_dict['8_mean_tree_depth'] = mean_tree_depth


        # 9. Average number of branches (dependency relations per word)
        def calc_branch_counts(dep_relation):
            heads = dep_relation['head']
            branch_counts = [0] * len(heads)
            for head in heads:
                if head > 0:
                    branch_counts[head - 1] += 1
            non_zero_branch_counts = [count for count in branch_counts if count > 0]
            return non_zero_branch_counts


        def calc_average_branch_count(dep_relation):
            branch_counts = calc_branch_counts(dep_relation)
            total_branch_count = sum(branch_counts)
            if len(branch_counts) > 0:
                average_branch_count = total_branch_count / len(branch_counts)
            else:
                average_branch_count = 0
            return average_branch_count


        branch_counts_per_sentence = [calc_average_branch_count(dep) for dep in dep_relations]
        mean_branch_count = np.mean(branch_counts_per_sentence)
        feature_dict['9_mean_branch_count'] = mean_branch_count

        # 10. Average frequency of attributive modifiers & 11. Average frequency of adverbial modifiers
        def calc_modifier_count(dep_relation, deprel_type):
            labels = dep_relation['label']
            return labels.count(deprel_type)

        attrib_counts = [calc_modifier_count(dep_rel, 'ATT') for dep_rel in dep_relations]  # Attributive modifiers
        advmod_counts = [calc_modifier_count(dep_rel, 'ADV') for dep_rel in dep_relations]  # Adverbial modifiers
        mean_attrib_count = np.mean(attrib_counts)
        mean_advmod_count = np.mean(advmod_counts)
        feature_dict['10_mean_attrib_count'] = mean_attrib_count
        feature_dict['11_mean_advmod_count'] = mean_advmod_count


    if textual_level:
        # 12. Ratio of positive sentences, 13. Ratio of neutral sentences, 14. Ratio of negative sentences
        def classify_sentence_sentiment(sentence):
            doc = nlp(sentence)
            sentiment_score = doc.sentences[0].sentiment
            if sentiment_score == 2:
                return 'positive'
            elif sentiment_score == 1:
                return 'neutral'
            else:
                return 'negative'

        sentiment_classification = [classify_sentence_sentiment(' '.join(words)) for words in segments]
        positive_sentence_ratio = sentiment_classification.count('positive') / total_sentences
        negative_sentence_ratio = sentiment_classification.count('negative') / total_sentences
        neutral_sentence_ratio = sentiment_classification.count('neutral') / total_sentences

        feature_dict['12_positive_sentence_ratio'] = positive_sentence_ratio
        feature_dict['13_negative_sentence_ratio'] = negative_sentence_ratio
        feature_dict['14_neutral_sentence_ratio'] = neutral_sentence_ratio

    # Add the feature dictionary to the list
    features.append(feature_dict)

    # Add the self-esteem column
    feature_dict['self-esteem'] = self_esteem
    participant_num += 1

# Save all features to a DataFrame
features_df = pd.DataFrame(features)

# Export the results to a new Excel file
features_df.to_excel(f'features.xlsx', index=False)
