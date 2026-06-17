import json
import re
import traceback
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Tuple


def load_evaluation_results(file_path: str) -> List[Dict]:
    """Load evaluation results from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded data type: {type(data)}")

    if isinstance(data, list):
        print(f"Found direct list with {len(data)} items")
        return data
    elif isinstance(data, dict):
        print(f"Found dictionary with keys: {list(data.keys())}")

        possible_keys = [
            'results', 'evaluation_results', 'data', 'items', 'evaluations',
            'retrieval_results', 'queries', 'questions', 'qa_results'
        ]

        for key in possible_keys:
            if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                print(f"Found results under key '{key}' with {len(data[key])} items")
                first_item = data[key][0]
                if isinstance(first_item, dict) and any(field in first_item for field in ['question', 'f1_score', 'correct_answer', 'predicted_items']):
                    return data[key]

        largest_list = None
        largest_key = None
        largest_size = 0

        for key, value in data.items():
            if isinstance(value, list) and len(value) > largest_size:
                if len(value) > 0 and isinstance(value[0], dict):
                    largest_list = value
                    largest_key = key
                    largest_size = len(value)

        if largest_list:
            print(f"Using largest list under key '{largest_key}' with {largest_size} items")
            return largest_list

        dict_values = list(data.values())
        if len(dict_values) > 10:
            sample_values = dict_values[:5]
            if all(isinstance(v, dict) for v in sample_values):
                if any(field in str(sample_values) for field in ['question', 'f1_score', 'correct_answer']):
                    print(f"Converting dictionary values to list ({len(dict_values)} items)")
                    return dict_values

        print("ERROR: Could not find evaluation results in the dictionary structure")
        print("Available keys:", list(data.keys())[:10])

        for key, value in list(data.items())[:3]:
            print(f"  {key}: {type(value)} {f'(length: {len(value)})' if hasattr(value, '__len__') else ''}")
            if isinstance(value, (dict, list)) and len(str(value)) < 200:
                print(f"    Content: {value}")

        return []
    else:
        print(f"ERROR: Unexpected data type: {type(data)}")
        return []


def convert_to_artem_format(evaluation_results: List[Dict]) -> Tuple[Dict, Dict]:
    """Convert evaluation results to ARTEM format for compatibility."""

    print(f"Converting {len(evaluation_results)} evaluation results...")

    retrieval_results = []
    ground_truth_data = {}
    skipped_count = 0

    for i, result in enumerate(evaluation_results):
        try:
            print(f"Processing result {i}: type={type(result)}")

            if isinstance(result, dict):
                print(f"  Keys: {list(result.keys())[:10]}...")

                question = result.get('question', '')
                if not question or question.strip() == '':
                    print(f"  WARNING: Skipping entry {i}: empty question")
                    skipped_count += 1
                    continue

                required_fields = ['question', 'correct_answer']
                missing_fields = [field for field in required_fields if field not in result]
                if missing_fields:
                    print(f"  WARNING: Skipping entry {i}: missing fields {missing_fields}")
                    skipped_count += 1
                    continue

                retrieved_events = result.get('retrieved_events', [])
                print(f"  retrieved_events type: {type(retrieved_events)}, length: {len(retrieved_events) if hasattr(retrieved_events, '__len__') else 'N/A'}")
            else:
                print(f"  ERROR: Result is not a dict: {result}")
                skipped_count += 1
                continue

            retrieval_result = {
                'query_id': result.get('q_idx', len(retrieval_results)),
                'query_text': result.get('question', ''),
                'results': {
                    'all_vigilant_events_time_sorted': result.get('retrieved_events', [])
                }
            }
            retrieval_results.append(retrieval_result)

            question_type = result.get('question_type', 'all')
            if question_type == 'all':
                get_type = 'all'
            elif question_type == 'latest':
                get_type = 'latest'
            elif question_type == 'chronological':
                get_type = 'chronological'
            else:
                get_type = 'all'

            ground_truth_item = {
                'question': result.get('question', ''),
                'correct_answer': result.get('correct_answer', []),
                'retrieval_type': result.get('retrieval_type', 'entities'),
                'get': get_type,
                'bins_items_correct_answer': get_bin(result.get('bins_items_correct_answer'))
            }
            ground_truth_data[len(retrieval_results)-1] = ground_truth_item

            print(f"  Processed successfully")

        except Exception as convert_error:
            print(f"ERROR converting result {i}: {convert_error}")
            traceback.print_exc()
            skipped_count += 1
            continue

    retrieval_data = {'retrieval_results': retrieval_results}

    print(f"Conversion complete: {len(retrieval_results)} retrieval results, {len(ground_truth_data)} ground truth items")
    if skipped_count > 0:
        print(f"WARNING: Skipped {skipped_count} problematic entries")

    return retrieval_data, ground_truth_data


def get_bin(n_correct: int) -> str:
    """Determine the bin based on the number of correct answers."""
    if n_correct == "0":
        return "0"
    elif n_correct == "1":
        return "1"
    elif n_correct == "2":
        return "2"
    elif n_correct == "{3,4,5}":
        return "3-5"
    else:
        return "6+"


def extract_retrieved_answers(retrieval_result: Dict, get_type: str, target_field: str) -> List[str]:
    """Extract answers from retrieval results based on get_type and target_field with deduplication."""
    results = retrieval_result.get('results', {})

    if get_type == "all":
        events = results.get('all_vigilant_events_time_sorted', [])
    elif get_type == "chronological":
        events = results.get('all_vigilant_events_time_sorted', [])
    elif get_type == "latest":
        events = results.get('all_vigilant_events_time_sorted', [])
        events = events[-1:] if events else []
    else:
        events = []

    if not events:
        return []

    if not isinstance(events, list):
        return []

    extracted = []
    for i, event in enumerate(events):
        if isinstance(event, dict):
            if target_field == 'spaces':
                value = event.get('spaces', '')
            elif target_field == 'entities':
                value = event.get('entities', '')
            elif target_field == 'content':
                value = event.get('content', '')
            elif target_field == 'time':
                time_val = event.get('time', [])
                if isinstance(time_val, list) and time_val:
                    value = time_val[0]
                else:
                    value = str(time_val)
            else:
                value = ''
        elif isinstance(event, str):
            if target_field in ['entities', 'spaces', 'content']:
                value = event
            else:
                value = ''
        else:
            value = str(event) if event else ''

        if value and str(value).strip() and str(value).strip().lower() not in ["none", "null", ""]:
            extracted.append(str(value).strip())

    seen = set()
    deduplicated = []
    for value in extracted:
        if value not in seen:
            seen.add(value)
            deduplicated.append(value)

    return deduplicated


def calculate_f1_score(predicted: List[str], actual: List[str]) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score."""
    predicted_set = set(predicted)
    actual_set = set(actual)

    if len(predicted_set) == 0 and len(actual_set) == 0:
        return 1.0, 1.0, 1.0
    elif len(predicted_set) == 0:
        return 0.0, 0.0, 0.0
    elif len(actual_set) == 0:
        return 0.0, 1.0, 0.0

    true_positive = len(predicted_set.intersection(actual_set))
    precision = true_positive / len(predicted_set) if len(predicted_set) > 0 else 0.0
    recall = true_positive / len(actual_set) if len(actual_set) > 0 else 0.0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compare_retrieval_results(retrieval_data: Dict, ground_truth_data: Dict, evaluation_results: List[Dict]) -> pd.DataFrame:
    """Compare retrieval results with ground truth and calculate F1 scores."""
    results = []

    retrieval_results = retrieval_data.get('retrieval_results', [])

    for i, (retrieval_result, eval_result) in enumerate(zip(retrieval_results, evaluation_results)):
        try:
            if i not in ground_truth_data:
                print(f"Warning: Index {i} not found in ground truth (total GT entries: {len(ground_truth_data)})")
                continue

            gt_item = ground_truth_data[i]
            q_idx = retrieval_result.get('query_id')
            query_text = retrieval_result.get('query_text', '')
            get_type = gt_item.get('get', 'all')
            retrieval_type = gt_item.get('retrieval_type', 'Entities')
            correct_answer = gt_item.get('correct_answer', [])

            print(f"\nProcessing index {i}:")
            print(f"  GT get_type: {get_type}")
            print(f"  GT retrieval_type: {retrieval_type}")
            print(f"  GT correct_answer: {correct_answer}")
            print(f"  Retrieval query_id: {q_idx}")
            print(f"  Retrieval query_text: {query_text}")

            field_mapping = {
                'spaces': 'spaces',
                'entities': 'entities',
                'content': 'content',
                'times': 'time',
                'time': 'time',
                'event contents': 'content',
                'full event details': 'content',
                'other entities': 'entities'
            }

            retrieval_type_lower = retrieval_type.lower()
            target_field = field_mapping.get(retrieval_type_lower, retrieval_type_lower)
            print(f"  Target field: {target_field} (mapped from {retrieval_type} -> {retrieval_type_lower})")

            try:
                retrieved_answers = extract_retrieved_answers(retrieval_result, get_type, target_field)
                print(f"  Retrieved answers: {retrieved_answers}")
            except Exception as extract_error:
                print(f"  ERROR extracting retrieved answers: {extract_error}")
                retrieved_answers = []

            if 'f1_score_lenient' in eval_result and eval_result['f1_score_lenient'] is not None:
                f1 = eval_result['f1_score_lenient']
                precision = eval_result.get('precision_lenient', 0.0) if eval_result.get('precision_lenient') is not None else 0.0
                recall = eval_result.get('recall', 0.0) if eval_result.get('recall') is not None else 0.0
                print(f"  Using existing F1 Score: {f1:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})")
            else:
                precision, recall, f1 = calculate_f1_score(retrieved_answers, correct_answer)
                print(f"  Calculated F1 Score: {f1:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})")

            results.append({
                'index': i,
                'query_id': q_idx,
                'query_text': query_text,
                'question': gt_item.get('question', ''),
                'retrieval_type': retrieval_type,
                'get_type': get_type,
                'correct_answer': correct_answer,
                'retrieved_answer': retrieved_answers,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'n_correct': len(correct_answer),
                'n_retrieved': len(retrieved_answers),
                'n_match': len(set(retrieved_answers).intersection(set(correct_answer))),
                'gt_bin': gt_item.get('bins_items_correct_answer', str(len(correct_answer)))
            })

        except Exception as item_error:
            print(f"ERROR processing item {i}: {item_error}")
            traceback.print_exc()
            continue

    return pd.DataFrame(results)


def create_performance_table(results_df: pd.DataFrame, use_gt_bins: bool = True) -> pd.DataFrame:
    """Create a performance table grouped by answer-count bins."""

    if use_gt_bins and 'gt_bin' in results_df.columns:
        results_df['bin'] = results_df['gt_bin'].astype(str)
        print("Using ground truth bins from 'bins_items_correct_answer' field")
    else:
        def assign_bin(n_correct):
            if n_correct == 0:
                return "0"
            elif n_correct == 1:
                return "1"
            elif n_correct == 2:
                return "2"
            elif 3 <= n_correct <= 5:
                return "3-5"
            else:
                return "6+"

        results_df['bin'] = results_df['n_correct'].apply(assign_bin)
        print("Using calculated bins based on number of correct answers")

    performance_stats = results_df.groupby('bin').agg({
        'f1_score': ['mean', 'std', 'count'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std']
    }).round(3)

    performance_stats.columns = ['_'.join(col).strip() for col in performance_stats.columns.values]

    formatted_table = pd.DataFrame()

    all_bins = sorted(results_df['bin'].unique(), key=lambda x: (len(x), x))

    for bin_name in all_bins:
        if bin_name in performance_stats.index:
            mean_f1 = performance_stats.loc[bin_name, 'f1_score_mean']
            std_f1 = performance_stats.loc[bin_name, 'f1_score_std']
            count = int(performance_stats.loc[bin_name, 'f1_score_count'])

            if pd.isna(std_f1):
                std_f1 = 0.0

            formatted_table.loc['Retrieval', f'{bin_name} ({count})'] = f'{mean_f1:.2f}±{std_f1:.2f}'

    return formatted_table


def create_retrieval_type_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a performance table categorized by retrieval type."""

    results_df_normalized = results_df.copy()

    def normalize_retrieval_type(ret_type):
        if pd.isna(ret_type):
            return ret_type
        ret_type_str = str(ret_type)
        if ret_type_str.lower() == 'entities':
            return 'Entities'
        elif ret_type_str.lower() == 'spaces':
            return 'Spaces'
        elif ret_type_str.lower() == 'times':
            return 'Times'
        elif ret_type_str.lower() in ['event contents', 'event_contents']:
            return 'Event contents'
        elif ret_type_str.lower() in ['full event details', 'full_event_details']:
            return 'Full event details'
        elif ret_type_str.lower() in ['other entities', 'other_entities']:
            return 'Other entities'
        else:
            return ret_type_str

    results_df_normalized['retrieval_type'] = results_df_normalized['retrieval_type'].apply(normalize_retrieval_type)

    type_stats = results_df_normalized.groupby('retrieval_type').agg({
        'f1_score': ['mean', 'std', 'count'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std']
    }).round(3)

    type_stats.columns = ['_'.join(col).strip() for col in type_stats.columns.values]

    formatted_table = pd.DataFrame()

    retrieval_types = sorted(results_df_normalized['retrieval_type'].unique())

    for ret_type in retrieval_types:
        if ret_type in type_stats.index:
            mean_f1 = type_stats.loc[ret_type, 'f1_score_mean']
            std_f1 = type_stats.loc[ret_type, 'f1_score_std']
            count = int(type_stats.loc[ret_type, 'f1_score_count'])

            if pd.isna(std_f1):
                std_f1 = 0.0

            formatted_table.loc['Retrieval', f'{ret_type} ({count})'] = f'{mean_f1:.2f}±{std_f1:.2f}'

    return formatted_table


def create_get_type_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a performance table categorized by get type (all/latest/chronological)."""

    get_stats = results_df.groupby('get_type').agg({
        'f1_score': ['mean', 'std', 'count'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std']
    }).round(3)

    get_stats.columns = ['_'.join(col).strip() for col in get_stats.columns.values]

    formatted_table = pd.DataFrame()

    preferred_order = ['all', 'latest', 'chronological']
    get_types = results_df['get_type'].unique()

    sorted_get_types = []
    for pref_type in preferred_order:
        if pref_type in get_types:
            sorted_get_types.append(pref_type)

    for get_type in sorted(get_types):
        if get_type not in sorted_get_types:
            sorted_get_types.append(get_type)

    for get_type in sorted_get_types:
        if get_type in get_stats.index:
            mean_f1 = get_stats.loc[get_type, 'f1_score_mean']
            std_f1 = get_stats.loc[get_type, 'f1_score_std']
            count = int(get_stats.loc[get_type, 'f1_score_count'])

            if pd.isna(std_f1):
                std_f1 = 0.0

            formatted_table.loc['Retrieval', f'{get_type} ({count})'] = f'{mean_f1:.2f}±{std_f1:.2f}'

    return formatted_table


def create_recall_vs_chronological_comparison(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a comparison table for simple recall (all) vs chronological retrieval."""

    def categorize_get_type(get_type):
        if get_type == 'all':
            return 'Simple Recall'
        elif get_type == 'chronological':
            return 'Chronological'
        elif get_type == 'latest':
            return 'Latest Only'
        else:
            return 'Other'

    results_df_copy = results_df.copy()
    results_df_copy['recall_category'] = results_df_copy['get_type'].apply(categorize_get_type)

    category_stats = results_df_copy.groupby('recall_category').agg({
        'f1_score': ['mean', 'std', 'count'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std']
    }).round(3)

    category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns.values]

    formatted_table = pd.DataFrame()

    preferred_order = ['Simple Recall', 'Chronological', 'Latest Only', 'Other']
    categories = results_df_copy['recall_category'].unique()

    sorted_categories = []
    for pref_cat in preferred_order:
        if pref_cat in categories:
            sorted_categories.append(pref_cat)

    for category in sorted_categories:
        if category in category_stats.index:
            mean_f1 = category_stats.loc[category, 'f1_score_mean']
            std_f1 = category_stats.loc[category, 'f1_score_std']
            count = int(category_stats.loc[category, 'f1_score_count'])
            mean_precision = category_stats.loc[category, 'precision_mean']
            mean_recall = category_stats.loc[category, 'recall_mean']

            if pd.isna(std_f1):
                std_f1 = 0.0

            formatted_table.loc['F1', f'{category} ({count})'] = f'{mean_f1:.2f}±{std_f1:.2f}'
            formatted_table.loc['Precision', f'{category} ({count})'] = f'{mean_precision:.2f}'
            formatted_table.loc['Recall', f'{category} ({count})'] = f'{mean_recall:.2f}'

    return formatted_table


def create_detailed_recall_chronological_analysis(results_df: pd.DataFrame) -> Dict:
    """Create detailed analysis comparing simple recall vs chronological retrieval."""

    results_df_normalized = results_df.copy()

    def normalize_retrieval_type(ret_type):
        if pd.isna(ret_type):
            return ret_type
        ret_type_str = str(ret_type)
        if ret_type_str.lower() == 'entities':
            return 'Entities'
        elif ret_type_str.lower() == 'spaces':
            return 'Spaces'
        elif ret_type_str.lower() == 'times':
            return 'Times'
        elif ret_type_str.lower() in ['event contents', 'event_contents']:
            return 'Event contents'
        elif ret_type_str.lower() in ['full event details', 'full_event_details']:
            return 'Full event details'
        elif ret_type_str.lower() in ['other entities', 'other_entities']:
            return 'Other entities'
        else:
            return ret_type_str

    results_df_normalized['retrieval_type'] = results_df_normalized['retrieval_type'].apply(normalize_retrieval_type)

    simple_recall = results_df_normalized[results_df_normalized['get_type'] == 'all']
    chronological = results_df_normalized[results_df_normalized['get_type'] == 'chronological']

    analysis = {
        'simple_recall': {
            'count': len(simple_recall),
            'f1_mean': simple_recall['f1_score'].mean() if len(simple_recall) > 0 else 0,
            'f1_std': simple_recall['f1_score'].std() if len(simple_recall) > 0 else 0,
            'precision_mean': simple_recall['precision'].mean() if len(simple_recall) > 0 else 0,
            'recall_mean': simple_recall['recall'].mean() if len(simple_recall) > 0 else 0,
            'by_retrieval_type': {}
        },
        'chronological': {
            'count': len(chronological),
            'f1_mean': chronological['f1_score'].mean() if len(chronological) > 0 else 0,
            'f1_std': chronological['f1_score'].std() if len(chronological) > 0 else 0,
            'precision_mean': chronological['precision'].mean() if len(chronological) > 0 else 0,
            'recall_mean': chronological['recall'].mean() if len(chronological) > 0 else 0,
            'by_retrieval_type': {}
        }
    }

    if len(simple_recall) > 0:
        for ret_type in simple_recall['retrieval_type'].unique():
            subset = simple_recall[simple_recall['retrieval_type'] == ret_type]
            analysis['simple_recall']['by_retrieval_type'][ret_type] = {
                'count': len(subset),
                'f1_mean': subset['f1_score'].mean(),
                'precision_mean': subset['precision'].mean(),
                'recall_mean': subset['recall'].mean()
            }

    if len(chronological) > 0:
        for ret_type in chronological['retrieval_type'].unique():
            subset = chronological[chronological['retrieval_type'] == ret_type]
            analysis['chronological']['by_retrieval_type'][ret_type] = {
                'count': len(subset),
                'f1_mean': subset['f1_score'].mean(),
                'precision_mean': subset['precision'].mean(),
                'recall_mean': subset['recall'].mean()
            }

    analysis['difference'] = {
        'f1_diff': analysis['simple_recall']['f1_mean'] - analysis['chronological']['f1_mean'],
        'precision_diff': analysis['simple_recall']['precision_mean'] - analysis['chronological']['precision_mean'],
        'recall_diff': analysis['simple_recall']['recall_mean'] - analysis['chronological']['recall_mean']
    }

    return analysis


def is_dont_know_response(model_answer: str) -> bool:
    """Check if the model answer is essentially saying 'I don't know'."""
    if not model_answer or not isinstance(model_answer, str):
        return False

    clean_answer = model_answer.strip()
    clean_answer = re.sub(r'<think>.*?</think>', '', clean_answer, flags=re.DOTALL)
    clean_answer = clean_answer.strip()

    lower_answer = clean_answer.lower()

    dont_know_patterns = [
        r"i don'?t know",
        r"no relevant information",
        r"no information (?:is )?provided",
        r"(?:cannot|can'?t) (?:find|provide|determine|identify)",
        r"(?:unable|not able) to (?:find|provide|determine|identify)",
        r"no (?:relevant )?(?:events?|data|details|information)",
        r"none found",
        r"not (?:available|provided|given)",
        r"insufficient information",
        r"no such (?:events?|information|data)",
        r"cannot be determined",
        r"not enough information"
    ]

    for pattern in dont_know_patterns:
        if re.search(pattern, lower_answer):
            return True

    if len(clean_answer.split()) <= 15:
        uncertainty_words = ['unknown', 'unclear', 'uncertain', 'unsure', 'no', 'none', 'nothing']
        words = lower_answer.split()
        if any(word in words for word in uncertainty_words):
            content_words = ['located', 'place', 'area', 'building', 'room', 'street', 'city', 'country']
            if not any(word in words for word in content_words):
                return True

    return False


def correct_bin_zero_f1_scores(results_df: pd.DataFrame) -> pd.DataFrame:
    """Correct F1 scores for bin '0' cases where the model correctly says 'I don't know'."""
    corrected_df = results_df.copy()
    corrections_made = 0

    print("\n=== CORRECTING BIN '0' F1 SCORES ===")

    bin_zero_cases = corrected_df[corrected_df['gt_bin'] == '0'].copy()

    print(f"Found {len(bin_zero_cases)} cases in bin '0'")

    for idx in bin_zero_cases.index:
        row = corrected_df.loc[idx]

        if (len(row['correct_answer']) == 0 and row['f1_score'] == 0.0):
            if len(row['retrieved_answer']) == 0:
                corrected_df.loc[idx, 'f1_score'] = 1.0
                corrected_df.loc[idx, 'precision'] = 1.0
                corrected_df.loc[idx, 'recall'] = 1.0
                corrections_made += 1

    if corrections_made > 0:
        print(f"Made {corrections_made} corrections to bin '0' F1 scores")
        new_overall_f1 = corrected_df['f1_score'].mean()
        old_overall_f1 = results_df['f1_score'].mean()
        print(f"Overall F1 changed: {old_overall_f1:.3f} -> {new_overall_f1:.3f} (+{new_overall_f1-old_overall_f1:.3f})")
    else:
        print("No corrections needed")

    return corrected_df


def get_overall_f1_score(results_df: pd.DataFrame) -> float:
    """Calculate the overall F1 score across all queries."""
    if results_df.empty or 'f1_score' not in results_df.columns:
        print("WARNING: No F1 scores available to calculate overall score")
        return 0.0
    return results_df['f1_score'].mean()


def correct_bin_zero_f1_scores_with_model_answers(results_df: pd.DataFrame, evaluation_results: List[Dict]) -> pd.DataFrame:
    """Correct F1 scores for bin '0' cases using actual model answers."""
    corrected_df = results_df.copy()
    corrections_made = 0

    print("\n=== CORRECTING BIN '0' F1 SCORES WITH MODEL ANSWERS ===")

    question_to_answer = {}
    for eval_result in evaluation_results:
        if isinstance(eval_result, dict):
            question = eval_result.get('question', '')
            model_answer = eval_result.get('model_answer', '')
            if question and model_answer:
                question_to_answer[question] = model_answer

    print(f"Created mapping for {len(question_to_answer)} questions")

    bin_zero_cases = corrected_df[
        (corrected_df['gt_bin'] == '0') &
        (corrected_df['f1_score'] == 0.0)
    ].copy()

    print(f"Found {len(bin_zero_cases)} bin '0' cases with F1=0 to check")

    for idx in bin_zero_cases.index:
        row = corrected_df.loc[idx]
        question = row['question']

        if question in question_to_answer:
            model_answer = question_to_answer[question]

            if is_dont_know_response(model_answer):
                print(f"  Case {idx}: Detected 'don't know' response")
                corrected_df.loc[idx, 'f1_score'] = 1.0
                corrected_df.loc[idx, 'precision'] = 1.0
                corrected_df.loc[idx, 'recall'] = 1.0
                corrections_made += 1

    if corrections_made > 0:
        print(f"Made {corrections_made} corrections to bin '0' F1 scores")
        new_overall_f1 = corrected_df['f1_score'].mean()
        old_overall_f1 = results_df['f1_score'].mean()
        print(f"Overall F1 changed: {old_overall_f1:.3f} -> {new_overall_f1:.3f} (+{new_overall_f1-old_overall_f1:.3f})")
    else:
        print("No corrections needed")

    return corrected_df


def filter_for_paper_version(results_df: pd.DataFrame) -> pd.DataFrame:
    """Filter results for paper version to achieve specific bin distributions."""

    if 'gt_bin' not in results_df.columns:
        print("Warning: 'gt_bin' column not found. Cannot create paper version.")
        return results_df.copy()

    filtered_df = pd.DataFrame()

    target_counts = {
        '0': 150, 0: 150,
        '1': 150, 1: 150,
        '2': 90, 2: 90,
        '3-5': 98,
        '6+': 60
    }

    print(f"Creating paper version with target distributions:")
    print(f"Available bins in data: {sorted(results_df['gt_bin'].unique())}")

    for bin_val in results_df['gt_bin'].unique():
        bin_questions = results_df[results_df['gt_bin'] == bin_val].copy()

        if len(bin_questions) > 0:
            original_count = len(bin_questions)
            target_count = target_counts.get(bin_val, original_count)

            if target_count < original_count:
                sampled_questions = bin_questions.sample(n=target_count, random_state=42)
                filtered_df = pd.concat([filtered_df, sampled_questions], ignore_index=True)
                print(f"  Bin {bin_val}: {target_count} questions kept out of {original_count} (randomly sampled)")
            else:
                filtered_df = pd.concat([filtered_df, bin_questions], ignore_index=True)
                print(f"  Bin {bin_val}: All {original_count} questions kept (target: {target_count})")
        else:
            print(f"  Bin {bin_val}: No questions found")

    print(f"Total questions: {len(results_df)} -> {len(filtered_df)}")

    final_distribution = filtered_df['gt_bin'].value_counts().sort_index()
    print(f"Final paper version distribution:")
    for bin_val, count in final_distribution.items():
        target = target_counts.get(bin_val, 'N/A')
        print(f"  Bin {bin_val}: {count} questions (target: {target})")

    return filtered_df


def analyze_retrieval_performance(evaluation_file_path: str, output_prefix: str = "ARTEM"):
    """Main function to analyze retrieval performance from evaluation results."""

    results_df = None
    performance_table_bins = None
    performance_table_types = None
    performance_table_get_types = None
    recall_vs_chronological_table = None
    recall_chronological_analysis = None
    overall_f1 = None

    try:
        print("Step 1: Loading evaluation data...")
        evaluation_results = load_evaluation_results(evaluation_file_path)

        if not evaluation_results:
            print("ERROR: No evaluation results found in the file!")
            raise ValueError("No evaluation results found in the input file")

        print(f"Loaded {len(evaluation_results)} evaluation results")

        print("Step 2: Converting to ARTEM format...")
        retrieval_data, ground_truth_data = convert_to_artem_format(evaluation_results)
        print("Data converted successfully")

        print("Step 3: Comparing results...")
        results_df = compare_retrieval_results(retrieval_data, ground_truth_data, evaluation_results)
        print("Results compared successfully")

        print("Step 3.1: Correcting F1 scores for 'don't know' responses...")
        results_df = correct_bin_zero_f1_scores_with_model_answers(results_df, evaluation_results)
        print("F1 score corrections applied")

        print("Step 4: Calculating overall F1...")
        overall_f1 = get_overall_f1_score(results_df)
        print(f"Overall F1: {overall_f1:.3f}")

        print("Step 5: Creating performance tables...")
        performance_table_bins = create_performance_table(results_df)
        performance_table_types = create_retrieval_type_table(results_df)
        performance_table_get_types = create_get_type_table(results_df)

        print("Step 5.1: Creating recall vs chronological analysis...")
        recall_vs_chronological_table = create_recall_vs_chronological_comparison(results_df)
        recall_chronological_analysis = create_detailed_recall_chronological_analysis(results_df)
        print("Tables created successfully")

        print("\n" + "="*80)
        print("OVERALL RETRIEVAL PERFORMANCE")
        print("="*80)
        print(f"OVERALL F1 SCORE: {overall_f1:.3f}")
        print("="*80)

        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY BY BINS:")
        print("="*80)
        print(performance_table_bins.to_string())

        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY BY RETRIEVAL TYPE:")
        print("="*80)
        print(performance_table_types.to_string())

        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY BY GET TYPE:")
        print("="*80)
        print(performance_table_get_types.to_string())

        print("\n" + "="*80)
        print("SIMPLE RECALL VS CHRONOLOGICAL COMPARISON:")
        print("="*80)
        print(recall_vs_chronological_table.to_string())

        print("\n" + "="*80)
        print("DETAILED RECALL VS CHRONOLOGICAL ANALYSIS:")
        print("="*80)
        simple_stats = recall_chronological_analysis['simple_recall']
        chrono_stats = recall_chronological_analysis['chronological']
        diff_stats = recall_chronological_analysis['difference']

        print(f"Simple Recall (all): {simple_stats['count']} queries")
        print(f"  F1: {simple_stats['f1_mean']:.3f} +/- {simple_stats['f1_std']:.3f}")
        print(f"  Precision: {simple_stats['precision_mean']:.3f}")
        print(f"  Recall: {simple_stats['recall_mean']:.3f}")

        print(f"\nChronological: {chrono_stats['count']} queries")
        print(f"  F1: {chrono_stats['f1_mean']:.3f} +/- {chrono_stats['f1_std']:.3f}")
        print(f"  Precision: {chrono_stats['precision_mean']:.3f}")
        print(f"  Recall: {chrono_stats['recall_mean']:.3f}")

        print(f"\nDifference (Simple - Chronological):")
        print(f"  F1 diff: {diff_stats['f1_diff']:+.3f}")
        print(f"  Precision diff: {diff_stats['precision_diff']:+.3f}")
        print(f"  Recall diff: {diff_stats['recall_diff']:+.3f}")

        print(f"\n" + "="*80)
        print("DETAILED STATISTICS:")
        print("="*80)
        print(f"Overall F1 Score: {overall_f1:.3f}")
        print(f"Mean Precision: {results_df['precision'].mean():.3f}")
        print(f"Mean Recall: {results_df['recall'].mean():.3f}")
        print(f"F1 Standard Deviation: {results_df['f1_score'].std():.3f}")
        print(f"Total Queries: {len(results_df)}")

        print(f"\nDistribution by Retrieval Type:")
        type_counts = results_df['retrieval_type'].value_counts()
        for ret_type, count in type_counts.items():
            type_f1 = results_df[results_df['retrieval_type'] == ret_type]['f1_score'].mean()
            print(f"  {ret_type}: {count} queries (F1: {type_f1:.3f})")

        print(f"\nDistribution by Get Type:")
        get_counts = results_df['get_type'].value_counts()
        for get_type, count in get_counts.items():
            get_f1 = results_df[results_df['get_type'] == get_type]['f1_score'].mean()
            print(f"  {get_type}: {count} queries (F1: {get_f1:.3f})")

        if 'gt_bin' in results_df.columns:
            print(f"\nDistribution by Ground Truth Bins:")
            bin_counts = results_df['gt_bin'].value_counts().sort_index()
            for bin_val, count in bin_counts.items():
                bin_f1 = results_df[results_df['gt_bin'] == bin_val]['f1_score'].mean()
                print(f"  Bin {bin_val}: {count} queries (F1: {bin_f1:.3f})")

        print(f"\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"OVERALL F1 SCORE: {overall_f1:.3f}")
        print("="*80)

        print("\nStep 6: Creating paper version...")
        results_df_paper = filter_for_paper_version(results_df)
        overall_f1_paper = get_overall_f1_score(results_df_paper)
        print(f"Paper version created, F1: {overall_f1_paper:.3f}")

        print("Step 7: Creating paper performance tables...")
        performance_table_bins_paper = create_performance_table(results_df_paper)
        performance_table_types_paper = create_retrieval_type_table(results_df_paper)
        performance_table_get_types_paper = create_get_type_table(results_df_paper)

        print("Step 7.1: Creating paper recall vs chronological analysis...")
        recall_vs_chronological_table_paper = create_recall_vs_chronological_comparison(results_df_paper)
        recall_chronological_analysis_paper = create_detailed_recall_chronological_analysis(results_df_paper)
        print("All paper tables created successfully")

        print(f"\n" + "="*80)
        print("UPDATED FINAL SUMMARY")
        print("="*80)
        print(f"OVERALL F1 SCORE (Full): {overall_f1:.3f}")
        print(f"OVERALL F1 SCORE (Paper): {overall_f1_paper:.3f}")
        print("="*80)

        print(f"\nStep 8: Saving files with prefix '{output_prefix}'...")

        results_df.to_json(f'{output_prefix}_retrieval_results_analysis.json', orient='records', indent=4)
        performance_table_bins.to_json(f'{output_prefix}_performance_table_bins.json', orient='records', indent=4)
        performance_table_types.to_json(f'{output_prefix}_performance_table_types.json', orient='records', indent=4)
        performance_table_get_types.to_json(f'{output_prefix}_performance_table_get_types.json', orient='records', indent=4)

        recall_vs_chronological_table.to_json(f'{output_prefix}_recall_vs_chronological_table.json', orient='records', indent=4)
        with open(f'{output_prefix}_recall_chronological_analysis.json', 'w') as f:
            json.dump(recall_chronological_analysis, f, indent=4)

        results_df_paper.to_json(f'{output_prefix}_retrieval_results_analysis_paper.json', orient='records', indent=4)
        performance_table_bins_paper.to_json(f'{output_prefix}_performance_table_bins_paper.json', orient='records', indent=4)
        performance_table_types_paper.to_json(f'{output_prefix}_performance_table_types_paper.json', orient='records', indent=4)
        performance_table_get_types_paper.to_json(f'{output_prefix}_performance_table_get_types_paper.json', orient='records', indent=4)

        recall_vs_chronological_table_paper.to_json(f'{output_prefix}_recall_vs_chronological_table_paper.json', orient='records', indent=4)
        with open(f'{output_prefix}_recall_chronological_analysis_paper.json', 'w') as f:
            json.dump(recall_chronological_analysis_paper, f, indent=4)

        overall_summary = {
            'overall_f1_score': overall_f1,
            'mean_precision': results_df['precision'].mean(),
            'mean_recall': results_df['recall'].mean(),
            'f1_std': results_df['f1_score'].std(),
            'total_queries': len(results_df),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        overall_summary_paper = {
            'overall_f1_score': overall_f1_paper,
            'mean_precision': results_df_paper['precision'].mean(),
            'mean_recall': results_df_paper['recall'].mean(),
            'f1_std': results_df_paper['f1_score'].std(),
            'total_queries': len(results_df_paper),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        with open(f'{output_prefix}_overall_performance.json', 'w') as f:
            json.dump(overall_summary, f, indent=4)
        with open(f'{output_prefix}_overall_performance_paper.json', 'w') as f:
            json.dump(overall_summary_paper, f, indent=4)

        print("All files saved successfully")
        print(f"OVERALL F1 SCORE (Full Dataset): {overall_f1:.3f}")
        print(f"OVERALL F1 SCORE (Paper Version): {overall_f1_paper:.3f}")

        simple_recall_f1 = recall_chronological_analysis['simple_recall']['f1_mean']
        chronological_f1 = recall_chronological_analysis['chronological']['f1_mean']
        print(f"\nRECALL COMPARISON:")
        print(f"Simple Recall F1: {simple_recall_f1:.3f}")
        print(f"Chronological F1: {chronological_f1:.3f}")
        print(f"Difference (Simple - Chrono): {simple_recall_f1 - chronological_f1:+.3f}")

        return (results_df, performance_table_bins, performance_table_types, performance_table_get_types, overall_f1,
                results_df_paper, performance_table_bins_paper, performance_table_types_paper,
                performance_table_get_types_paper, overall_f1_paper,
                recall_vs_chronological_table, recall_chronological_analysis,
                recall_vs_chronological_table_paper, recall_chronological_analysis_paper)

    except Exception as e:
        print(f"\nERROR occurred during analysis: {e}")
        traceback.print_exc()

        if all(v is not None for v in [results_df, performance_table_bins, performance_table_types,
                                        performance_table_get_types, overall_f1]):
            print("Returning basic 5-value version due to error in paper version creation...")
            return results_df, performance_table_bins, performance_table_types, performance_table_get_types, overall_f1
        else:
            print("ERROR: Cannot return even basic version - critical error occurred")
            raise e


if __name__ == "__main__":
    evaluation_file_path = "bookname/book1/art_evaluation_results/art_stem_evaluation_books_1_detailed_results.json"
    output_prefix = "bookname/book1/art_evaluation_results/LLM_ARTEM"

    print("=== MAIN EXECUTION STARTING ===")

    try:
        results = analyze_retrieval_performance(evaluation_file_path, output_prefix)
        print("=== FUNCTION CALL COMPLETED ===")
        print(f"Function returned {len(results)} values")

        if len(results) == 14:
            print("Successfully generated all 14 output components!")
        else:
            print(f"WARNING: Generated {len(results)} components instead of expected 14.")

    except Exception as e:
        print(f"ERROR IN FUNCTION CALL: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
