import json
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Tuple


def load_ground_truth(file_path: str) -> Dict[int, Dict]:
    """Load ground truth data from JSON file."""
    with open(file_path, 'r') as f:
        ground_truth_list = json.load(f)

    ground_truth = {i: item for i, item in enumerate(ground_truth_list)}
    return ground_truth


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

    extracted = []
    for event in events:
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

        if value and str(value).strip() and str(value).strip() != "none":
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


def compare_retrieval_results(retrieval_data: Dict, ground_truth_data: Dict) -> pd.DataFrame:
    """Compare retrieval results with ground truth and calculate F1 scores."""
    results = []

    retrieval_results = retrieval_data.get('retrieval_results', [])

    for i, retrieval_result in enumerate(retrieval_results):
        if i not in ground_truth_data:
            print(f"Warning: Index {i} not found in ground truth (total GT entries: {len(ground_truth_data)})")
            continue

        gt_item = ground_truth_data[i]
        q_idx = retrieval_result.get('query_id')
        query_text = retrieval_result.get('query_text', '')
        get_type = gt_item.get('get', 'all')
        retrieval_type = gt_item.get('retrieval_type', 'Spaces')
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
            'event contents': 'content'
        }
        target_field = field_mapping.get(retrieval_type.lower(), retrieval_type.lower())
        print(f"  Target field: {target_field} (mapped from {retrieval_type})")

        retrieved_answers = extract_retrieved_answers(retrieval_result, get_type, target_field)
        print(f"  Retrieved answers: {retrieved_answers}")

        precision, recall, f1 = calculate_f1_score(retrieved_answers, correct_answer)
        print(f"  F1 Score: {f1:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})")

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

    return pd.DataFrame(results)


def create_performance_table(results_df: pd.DataFrame, use_gt_bins: bool = True) -> pd.DataFrame:
    """Create a performance table grouped by answer-count bins."""
    if use_gt_bins:
        gt_json_path = "data_root/book1/qa_book1.json"

        try:
            with open(gt_json_path, 'r') as f:
                gt_data = json.load(f)

            gt_bins_map = {idx: item['bins_items_correct_answer'] for idx, item in enumerate(gt_data)}
            results_df['bin'] = results_df.index.map(gt_bins_map).astype(str)
            results_df['bin'] = results_df['bin'].fillna('unknown')

        except FileNotFoundError:
            print(f"Warning: Ground truth file not found at {gt_json_path}")
            print("Falling back to existing bin assignment method")
            if 'gt_bin' in results_df.columns:
                results_df['bin'] = results_df['gt_bin'].astype(str)
        except Exception as e:
            print(f"Error loading ground truth bins: {e}")
            if 'gt_bin' in results_df.columns:
                results_df['bin'] = results_df['gt_bin'].astype(str)

    elif use_gt_bins and 'gt_bin' in results_df.columns:
        results_df['bin'] = results_df['gt_bin'].astype(str)

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
    type_stats = results_df.groupby('retrieval_type').agg({
        'f1_score': ['mean', 'std', 'count'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std']
    }).round(3)

    type_stats.columns = ['_'.join(col).strip() for col in type_stats.columns.values]

    formatted_table = pd.DataFrame()

    retrieval_types = sorted(results_df['retrieval_type'].unique())

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

    simple_recall = results_df[results_df['get_type'] == 'all']
    chronological = results_df[results_df['get_type'] == 'chronological']

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


def get_overall_f1_score(results_df: pd.DataFrame) -> float:
    """Calculate the overall F1 score across all queries."""
    return results_df['f1_score'].mean()


def filter_for_paper_version(results_df: pd.DataFrame) -> pd.DataFrame:
    """Filter results for paper version by removing specified questions from each bin."""

    if 'gt_bin' not in results_df.columns:
        print("Warning: 'gt_bin' column not found. Cannot create paper version.")
        return results_df.copy()

    filtered_df = results_df.copy()

    remove_counts = {
        '0': 30, 0: 30,
        '1': 30, 1: 30,
        '2': 26, 2: 26
    }

    print(f"Creating paper version by removing questions from bins:")
    print(f"Available bins in data: {sorted(results_df['gt_bin'].unique())}")

    for bin_val in results_df['gt_bin'].unique():
        bin_questions = filtered_df[filtered_df['gt_bin'] == bin_val]

        if len(bin_questions) > 0:
            original_count = len(bin_questions)
            remove_count = remove_counts.get(bin_val, 0)

            if remove_count > 0:
                questions_to_remove = min(remove_count, original_count)
                indices_to_remove = bin_questions.head(questions_to_remove).index
                filtered_df = filtered_df.drop(indices_to_remove)

                remaining_count = len(filtered_df[filtered_df['gt_bin'] == bin_val])
                print(f"  Bin {bin_val}: {questions_to_remove} questions removed ({original_count} -> {remaining_count})")
            else:
                print(f"  Bin {bin_val}: No questions removed ({original_count} questions)")
        else:
            print(f"  Bin {bin_val}: No questions found")

    print(f"Total questions: {len(results_df)} -> {len(filtered_df)}")
    return filtered_df


def analyze_retrieval_performance(retrieval_file_path: str, ground_truth_file_path: str):
    """Main function to analyze retrieval performance."""

    results_df = None
    performance_table_bins = None
    performance_table_types = None
    performance_table_get_types = None
    recall_vs_chronological_table = None
    recall_chronological_analysis = None
    overall_f1 = None

    try:
        print("Step 1: Loading data...")
        with open(retrieval_file_path, 'r') as f:
            retrieval_data = json.load(f)

        ground_truth_data = load_ground_truth(ground_truth_file_path)
        print("Data loaded successfully")

        print("Step 2: Comparing results...")
        results_df = compare_retrieval_results(retrieval_data, ground_truth_data)
        print("Results compared successfully")

        print("Step 3: Calculating overall F1...")
        overall_f1 = get_overall_f1_score(results_df)
        print(f"Overall F1: {overall_f1:.3f}")

        print("Step 4: Creating performance tables...")
        performance_table_bins = create_performance_table(results_df)
        performance_table_types = create_retrieval_type_table(results_df)
        performance_table_get_types = create_get_type_table(results_df)

        print("Step 4.1: Creating recall vs chronological analysis...")
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

        print("\nStep 5: Creating paper version...")
        results_df_paper = filter_for_paper_version(results_df)
        overall_f1_paper = get_overall_f1_score(results_df_paper)
        print(f"Paper version created, F1: {overall_f1_paper:.3f}")

        print("Step 6: Creating paper performance tables...")
        performance_table_bins_paper = create_performance_table(results_df_paper)
        performance_table_types_paper = create_retrieval_type_table(results_df_paper)
        performance_table_get_types_paper = create_get_type_table(results_df_paper)

        print("Step 6.1: Creating paper recall vs chronological analysis...")
        recall_vs_chronological_table_paper = create_recall_vs_chronological_comparison(results_df_paper)
        recall_chronological_analysis_paper = create_detailed_recall_chronological_analysis(results_df_paper)

        print(f"\n" + "="*80)
        print("UPDATED FINAL SUMMARY")
        print("="*80)
        print(f"OVERALL F1 SCORE (Full): {overall_f1:.3f}")
        print(f"OVERALL F1 SCORE (Paper): {overall_f1_paper:.3f}")
        print("="*80)

        return (results_df, performance_table_bins, performance_table_types, performance_table_get_types, overall_f1,
                results_df_paper, performance_table_bins_paper, performance_table_types_paper,
                performance_table_get_types_paper, overall_f1_paper,
                recall_vs_chronological_table, recall_chronological_analysis,
                recall_vs_chronological_table_paper, recall_chronological_analysis_paper)

    except Exception as e:
        print(f"\nERROR occurred during paper version creation: {e}")
        import traceback
        traceback.print_exc()

        if all(v is not None for v in [results_df, performance_table_bins, performance_table_types,
                                        performance_table_get_types, overall_f1]):
            print("Returning basic 5-value version due to error in paper version creation...")
            return results_df, performance_table_bins, performance_table_types, performance_table_get_types, overall_f1
        else:
            print("ERROR: Cannot return even basic version - critical error occurred")
            raise e


if __name__ == "__main__":
    print("=== MAIN EXECUTION STARTING ===")

    try:
        results = analyze_retrieval_performance(
            'data_root/book1/match_based_retrieval_results_book1.json',
            'data_root/book1/qa_book1.json'
        )
        print("=== FUNCTION CALL COMPLETED ===")
    except Exception as e:
        print(f"ERROR IN FUNCTION CALL: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print(f"Function returned {len(results)} values")

    if len(results) == 14:
        (results_df, performance_table_bins, performance_table_types, performance_table_get_types, overall_f1,
         results_df_paper, performance_table_bins_paper, performance_table_types_paper,
         performance_table_get_types_paper, overall_f1_paper,
         recall_vs_chronological_table, recall_chronological_analysis,
         recall_vs_chronological_table_paper, recall_chronological_analysis_paper) = results

        print("Successfully unpacked 14 values!")

        results_df.to_json('data_root/book1/ARTEM_retrieval_results_analysis_book1.json', orient='records', indent=4)
        performance_table_bins.to_json('data_root/book1/ARTEM_performance_table_bins_book1.json', orient='records', indent=4)
        performance_table_types.to_json('data_root/book1/ARTEM_performance_table_types_book1.json', orient='records', indent=4)
        performance_table_get_types.to_json('data_root/book1/ARTEM_performance_table_get_types_book1.json', orient='records', indent=4)

        recall_vs_chronological_table.to_json('data_root/book1/ARTEM_recall_vs_chronological_table_book1.json', orient='records', indent=4)
        with open('data_root/book1/ARTEM_recall_chronological_analysis_book1.json', 'w') as f:
            json.dump(recall_chronological_analysis, f, indent=4)

        results_df_paper.to_json('data_root/book1/ARTEM_retrieval_results_analysis_book1_paper.json', orient='records', indent=4)
        performance_table_bins_paper.to_json('data_root/book1/ARTEM_performance_table_bins_book1_paper.json', orient='records', indent=4)
        performance_table_types_paper.to_json('data_root/book1/ARTEM_performance_table_types_book1_paper.json', orient='records', indent=4)
        performance_table_get_types_paper.to_json('data_root/book1/ARTEM_performance_table_get_types_book1_paper.json', orient='records', indent=4)

        recall_vs_chronological_table_paper.to_json('data_root/book1/ARTEM_recall_vs_chronological_table_book1_paper.json', orient='records', indent=4)
        with open('data_root/book1/ARTEM_recall_chronological_analysis_book1_paper.json', 'w') as f:
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

        with open('data_root/book1/ARTEM_overall_performance_book1.json', 'w') as f:
            json.dump(overall_summary, f, indent=4)
        with open('data_root/book1/ARTEM_overall_performance_book1_paper.json', 'w') as f:
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

    elif len(results) == 5:
        results_df, performance_table_bins, performance_table_types, performance_table_get_types, overall_f1 = results
        print(f"Only saving original version with F1: {overall_f1:.3f}")

        results_df.to_json('data_root/book1/ARTEM_retrieval_results_analysis_book1.json', orient='records', indent=4)
        performance_table_bins.to_json('data_root/book1/ARTEM_performance_table_bins_book1.json', orient='records', indent=4)
        performance_table_types.to_json('data_root/book1/ARTEM_performance_table_types_book1.json', orient='records', indent=4)
        performance_table_get_types.to_json('data_root/book1/ARTEM_performance_table_get_types_book1.json', orient='records', indent=4)

        print("Original version files saved.")

    else:
        print(f"ERROR: Unexpected number of return values: {len(results)}")

    print("=== MAIN EXECUTION COMPLETED ===")
