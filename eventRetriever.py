import os
import re
import json
import math
import datetime
import copy
import traceback
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse

from eventNormalizer import EventNormalizer
from fusionART import *
from ARTxtralib import *
from eventProcessor import process_event_data


def convert_np(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def retrieve_all_above_vigilance_with_match(fa, query_vector, gamma, vectorized_events,
                                            rho_threshold=[1.0, 0.99, 0.995, 0.98],
                                            max_retrievals=10, return_all_vigilant=True):
    """
    Retrieve events based on match values (compatibility) instead of activation scores.
    """
    all_vigilant_events = []
    all_weighted_matches = []
    all_match_scores = []

    fa.setActivityF1(query_vector)

    print(f"Activity F1 set:")
    for k, qv in enumerate(query_vector):
        if isinstance(qv, list) and len(qv) > 0:
            qv_array = np.array(qv)
            print(f"  Channel {k}: shape={qv_array.shape}, norm={np.linalg.norm(qv_array):.4f}, gamma={gamma[k]}")
            if gamma[k] > 0 and np.linalg.norm(qv_array) == 0:
                print(f"    WARNING: Active channel {k} has zero norm - may cause issues")

    active_channels = [k for k in range(len(gamma)) if gamma[k] > 0]
    time_channel_active = 0 in active_channels

    print(f"Match-based retrieval analysis:")
    print(f"  Active channels (gamma > 0): {active_channels}")
    print(f"  Gamma weights: {gamma}")
    print(f"  Total codes: {len(fa.codes)}")

    if time_channel_active:
        print(f"TIME QUERY DETECTED!")
        print(f"  Query time: {query_vector[0][0]:.8f}")
        print(f"  Time vigilance threshold: {rho_threshold[0]}")

        sample_stored_times = []
        for i in range(min(20, len(vectorized_events))):
            if vectorized_events[i][0] and len(vectorized_events[i][0]) > 0:
                sample_stored_times.append(vectorized_events[i][0][0])

        if sample_stored_times:
            unique_times = len(set([f"{t:.8f}" for t in sample_stored_times]))
            print(f"  Sample stored times: {[f'{t:.8f}' for t in sample_stored_times[:5]]}")
            print(f"  Unique stored times in sample: {unique_times}/20")

            if unique_times < 5:
                print(f"  WARNING: Very few unique time values - possible normalization issue!")
                print(f"  Time values: {sorted(set([f'{t:.8f}' for t in sample_stored_times]))}")

    for k, qv in enumerate(query_vector):
        if isinstance(qv, list) and len(qv) > 0:
            qv_array = np.array(qv)
            if np.any(np.isnan(qv_array)):
                print(f"    WARNING: NaN detected in query channel {k}")
            if np.any(np.isinf(qv_array)):
                print(f"    WARNING: Inf detected in query channel {k}")
            if gamma[k] > 0 and np.linalg.norm(qv_array) == 0:
                print(f"    WARNING: Zero norm in active channel {k} - this may cause match issues")

    vigilance_passed = 0
    vigilance_failed = 0
    time_match_scores = []

    for j in range(len(fa.codes)):
        if not fa.uncommitted(j):
            matches = [fa.matchValField[k](query_vector[k], fa.codes[j]['weights'][k])
                       for k in range(len(query_vector))]

            if any(np.isnan(m) for m in matches):
                print(f"WARNING: NaN in matches for node {j}: {matches}")
                continue

            all_match_scores.extend(matches)

            if time_channel_active:
                time_match_scores.append(matches[0])

            if time_channel_active and j < 10:
                stored_time_vector = fa.codes[j]['weights'][0]
                query_time_vector = query_vector[0]
                time_match_score = matches[0]

                original_stored_time = vectorized_events[j][0][0] if j < len(vectorized_events) else 'N/A'
                time_diff = abs(original_stored_time - query_vector[0][0]) if original_stored_time != 'N/A' else 'N/A'

                print(f"  Node {j}: stored_time={original_stored_time:.8f}, query_time={query_vector[0][0]:.8f}")
                print(f"           time_diff={time_diff:.8f}, match_score={time_match_score:.8f}")

                if time_match_score >= rho_threshold[0]:
                    print(f"           PASSES vigilance ({time_match_score:.8f} >= {rho_threshold[0]})")
                else:
                    print(f"           FAILS vigilance ({time_match_score:.8f} < {rho_threshold[0]})")

            avg_match = float(np.mean(matches))

            if active_channels:
                weighted_match = sum(gamma[k] * matches[k] for k in active_channels) / sum(gamma[k] for k in active_channels)
            else:
                weighted_match = avg_match

            all_weighted_matches.append(weighted_match)

            min_match = float(np.min(matches))
            max_match = float(np.max(matches))

            normalized_time = 0.0
            if j < len(vectorized_events):
                time_channel = vectorized_events[j][0]
                if time_channel and len(time_channel) > 0:
                    normalized_time = float(time_channel[0])
                normalized_time = max(0, min(1, normalized_time))

            event_data = {
                'node_id': j + 1,
                'primary_score': float(weighted_match),
                'match_scores': [float(m) for m in matches],
                'avg_match': avg_match,
                'weighted_match': float(weighted_match),
                'min_match': min_match,
                'max_match': max_match,
                'normalized_time': normalized_time,
                'vigilance_passed': False,
                'active_channels': active_channels,
                'gamma_weights': [gamma[k] for k in range(len(gamma))]
            }

            channel_passes = []
            all_channels_pass = True

            for k in range(len(matches)):
                channel_passes_vigilance = matches[k] >= rho_threshold[k]
                channel_passes.append({
                    'channel': k,
                    'match_score': float(matches[k]),
                    'vigilance_threshold': rho_threshold[k],
                    'passed': channel_passes_vigilance
                })

                if gamma[k] > 0:
                    if not channel_passes_vigilance:
                        all_channels_pass = False
                        break

            event_data['channel_vigilance_results'] = channel_passes

            if all_channels_pass:
                event_data['vigilance_passed'] = True
                all_vigilant_events.append(event_data)
                vigilance_passed += 1
            else:
                vigilance_failed += 1

    if time_channel_active and time_match_scores:
        print(f"TIME QUERY ANALYSIS:")
        print(f"  Total time match scores: {len(time_match_scores)}")
        print(f"  Time match score range: [{min(time_match_scores):.8f}, {max(time_match_scores):.8f}]")
        print(f"  Events passing vigilance: {vigilance_passed}")

        if vigilance_passed > 20:
            print(f"  WARNING: Too many events passing - likely normalization issue!")

            time_diffs = []
            query_time = query_vector[0][0]
            for i in range(min(20, len(vectorized_events))):
                stored_time = vectorized_events[i][0][0] if vectorized_events[i][0] else 0.0
                time_diffs.append(abs(stored_time - query_time))

            print(f"  Time differences from query: min={min(time_diffs):.8f}, max={max(time_diffs):.8f}")

    print(f"Vigilance Results: {vigilance_passed} passed, {vigilance_failed} failed (threshold={rho_threshold})")

    all_vigilant_events.sort(key=lambda x: x['weighted_match'], reverse=True)

    top_k_events = all_vigilant_events[:max_retrievals]
    highest_scoring_event = all_vigilant_events[0] if all_vigilant_events else None

    vigilance_stats = {
        'total_nodes_checked': len(fa.codes) - sum(1 for j in range(len(fa.codes)) if fa.uncommitted(j)),
        'vigilance_passed': vigilance_passed,
        'vigilance_failed': vigilance_failed,
        'vigilance_pass_rate': vigilance_passed / (vigilance_passed + vigilance_failed) if (vigilance_passed + vigilance_failed) > 0 else 0,
        'vigilance_threshold': rho_threshold,
        'time_channel_active': time_channel_active,
        'time_match_analysis': {
            'total_scores': len(time_match_scores) if time_channel_active else 0,
            'perfect_matches': sum(1 for s in time_match_scores if s >= 1.0) if time_channel_active else 0,
            'near_perfect_matches': sum(1 for s in time_match_scores if s >= 0.99) if time_channel_active else 0,
            'score_range': [min(time_match_scores), max(time_match_scores)] if time_match_scores else [0, 0]
        },
        'all_weighted_matches': {
            'min': float(min(all_weighted_matches)) if all_weighted_matches else 0,
            'max': float(max(all_weighted_matches)) if all_weighted_matches else 0,
            'mean': float(np.mean(all_weighted_matches)) if all_weighted_matches else 0,
            'std': float(np.std(all_weighted_matches)) if all_weighted_matches else 0,
            'count': len(all_weighted_matches)
        },
        'all_match_scores': {
            'min': float(min(all_match_scores)) if all_match_scores else 0,
            'max': float(max(all_match_scores)) if all_match_scores else 0,
            'mean': float(np.mean(all_match_scores)) if all_match_scores else 0,
            'std': float(np.std(all_match_scores)) if all_match_scores else 0,
            'count': len(all_match_scores)
        },
        'passing_weighted_matches': {
            'min': float(min([e['weighted_match'] for e in all_vigilant_events])) if all_vigilant_events else 0,
            'max': float(max([e['weighted_match'] for e in all_vigilant_events])) if all_vigilant_events else 0,
            'mean': float(np.mean([e['weighted_match'] for e in all_vigilant_events])) if all_vigilant_events else 0,
            'count': len(all_vigilant_events)
        }
    }

    print(f"Vigilance Stats: pass rate {vigilance_stats['vigilance_pass_rate']:.2%} "
          f"({vigilance_passed}/{vigilance_passed + vigilance_failed})")

    return {
        'top_k_events': top_k_events,
        'all_vigilant_events': all_vigilant_events if return_all_vigilant else [],
        'highest_scoring_event': highest_scoring_event,
        'vigilance_stats': vigilance_stats
    }


def retrieve_all_above_vigilance_with_exact_time_match(fa, query_vector, gamma, vectorized_events,
                                                       rho_threshold=[1.0, 0.99, 0.995, 0.98],
                                                       max_retrievals=10, return_all_vigilant=True):
    """
    Retrieve events based on match values with exact time matching for time queries.
    """
    all_vigilant_events = []
    all_weighted_matches = []
    all_match_scores = []

    fa.setActivityF1(query_vector)

    print(f"Activity F1 set:")
    for k, qv in enumerate(query_vector):
        if isinstance(qv, list) and len(qv) > 0:
            qv_array = np.array(qv)
            print(f"  Channel {k}: shape={qv_array.shape}, norm={np.linalg.norm(qv_array):.4f}, gamma={gamma[k]}")
            if gamma[k] > 0 and np.linalg.norm(qv_array) == 0:
                print(f"    WARNING: Active channel {k} has zero norm - may cause issues")

    active_channels = [k for k in range(len(gamma)) if gamma[k] > 0]
    time_channel_active = 0 in active_channels

    print(f"Match-based retrieval analysis:")
    print(f"  Active channels (gamma > 0): {active_channels}")
    print(f"  Gamma weights: {gamma}")
    print(f"  Total codes: {len(fa.codes)}")

    if time_channel_active:
        print(f"TIME QUERY DETECTED - USING EXACT TIME MATCHING!")
        print(f"  Query time: {query_vector[0][0]:.8f}")
        print(f"  Time vigilance threshold: {rho_threshold[0]} (will be overridden for exact matching)")

        sample_stored_times = []
        for i in range(min(20, len(vectorized_events))):
            if vectorized_events[i][0] and len(vectorized_events[i][0]) > 0:
                sample_stored_times.append(vectorized_events[i][0][0])

        if sample_stored_times:
            unique_times = len(set([f"{t:.8f}" for t in sample_stored_times]))
            print(f"  Sample stored times: {[f'{t:.8f}' for t in sample_stored_times[:5]]}")
            print(f"  Unique stored times in sample: {unique_times}/20")

            exact_time_matches = sum(1 for t in sample_stored_times if abs(t - query_vector[0][0]) < 1e-10)
            print(f"  Events with exactly matching time (in sample): {exact_time_matches}/20")

    for k, qv in enumerate(query_vector):
        if isinstance(qv, list) and len(qv) > 0:
            qv_array = np.array(qv)
            if np.any(np.isnan(qv_array)):
                print(f"    WARNING: NaN detected in query channel {k}")
            if np.any(np.isinf(qv_array)):
                print(f"    WARNING: Inf detected in query channel {k}")
            if gamma[k] > 0 and np.linalg.norm(qv_array) == 0:
                print(f"    WARNING: Zero norm in active channel {k} - this may cause match issues")

    vigilance_passed = 0
    vigilance_failed = 0
    time_match_scores = []
    exact_time_matches_found = 0

    for j in range(len(fa.codes)):
        if not fa.uncommitted(j):
            if time_channel_active:
                if j < len(vectorized_events):
                    stored_time = vectorized_events[j][0][0] if vectorized_events[j][0] else 0.0
                    query_time = query_vector[0][0]

                    time_epsilon = 1e-10
                    time_matches_exactly = abs(stored_time - query_time) < time_epsilon

                    if not time_matches_exactly:
                        if j < 10:
                            print(f"  Node {j}: REJECTED - time mismatch: stored={stored_time:.8f}, "
                                  f"query={query_time:.8f}, diff={abs(stored_time - query_time):.8f}")
                        continue
                    else:
                        exact_time_matches_found += 1
                        if j < 10:
                            print(f"  Node {j}: ACCEPTED - exact time match: stored={stored_time:.8f}, query={query_time:.8f}")

            matches = [fa.matchValField[k](query_vector[k], fa.codes[j]['weights'][k])
                       for k in range(len(query_vector))]

            if any(np.isnan(m) for m in matches):
                print(f"WARNING: NaN in matches for node {j}: {matches}")
                continue

            all_match_scores.extend(matches)

            if time_channel_active:
                time_match_scores.append(matches[0])

            avg_match = float(np.mean(matches))

            if active_channels:
                weighted_match = sum(gamma[k] * matches[k] for k in active_channels) / sum(gamma[k] for k in active_channels)
            else:
                weighted_match = avg_match

            all_weighted_matches.append(weighted_match)

            min_match = float(np.min(matches))
            max_match = float(np.max(matches))

            normalized_time = 0.0
            if j < len(vectorized_events):
                time_channel = vectorized_events[j][0]
                if time_channel and len(time_channel) > 0:
                    normalized_time = float(time_channel[0])

            event_data = {
                'node_id': j + 1,
                'primary_score': float(weighted_match),
                'match_scores': [float(m) for m in matches],
                'avg_match': avg_match,
                'weighted_match': float(weighted_match),
                'min_match': min_match,
                'max_match': max_match,
                'normalized_time': normalized_time,
                'vigilance_passed': False,
                'active_channels': active_channels,
                'gamma_weights': [gamma[k] for k in range(len(gamma))],
                'exact_time_match': True if time_channel_active else None
            }

            channel_passes = []
            all_channels_pass = True

            for k in range(len(matches)):
                if k == 0 and time_channel_active:
                    channel_passes_vigilance = True
                else:
                    channel_passes_vigilance = matches[k] >= rho_threshold[k]

                channel_passes.append({
                    'channel': k,
                    'match_score': float(matches[k]),
                    'vigilance_threshold': rho_threshold[k],
                    'passed': channel_passes_vigilance,
                    'exact_time_match': k == 0 and time_channel_active
                })

                if gamma[k] > 0 and not channel_passes_vigilance:
                    all_channels_pass = False

            event_data['channel_vigilance_results'] = channel_passes

            if all_channels_pass:
                event_data['vigilance_passed'] = True
                all_vigilant_events.append(event_data)
                vigilance_passed += 1
            else:
                vigilance_failed += 1

    if time_channel_active:
        print(f"EXACT TIME QUERY ANALYSIS:")
        print(f"  Exact time matches found: {exact_time_matches_found}")
        print(f"  Events passing all vigilance: {vigilance_passed}")
        print(f"  Query time: {query_vector[0][0]:.8f}")

        if exact_time_matches_found == 0:
            print(f"  WARNING: No exact time matches found!")
        else:
            print(f"  Found {exact_time_matches_found} events with exact time match")
            if vigilance_passed < exact_time_matches_found:
                print(f"  Note: {exact_time_matches_found - vigilance_passed} exact time matches failed other channel vigilance")

    print(f"Exact Time Match-based Vigilance Results: {vigilance_passed} passed, {vigilance_failed} failed")

    all_vigilant_events.sort(key=lambda x: x['weighted_match'], reverse=True)

    top_k_events = all_vigilant_events[:max_retrievals]
    highest_scoring_event = all_vigilant_events[0] if all_vigilant_events else None

    vigilance_stats = {
        'total_nodes_checked': len(fa.codes) - sum(1 for j in range(len(fa.codes)) if fa.uncommitted(j)),
        'vigilance_passed': vigilance_passed,
        'vigilance_failed': vigilance_failed,
        'vigilance_pass_rate': vigilance_passed / (vigilance_passed + vigilance_failed) if (vigilance_passed + vigilance_failed) > 0 else 0,
        'vigilance_threshold': rho_threshold,
        'time_channel_active': time_channel_active,
        'exact_time_matching_enabled': time_channel_active,
        'exact_time_matches_found': exact_time_matches_found if time_channel_active else 0,
        'time_match_analysis': {
            'total_scores': len(time_match_scores) if time_channel_active else 0,
            'perfect_matches': sum(1 for s in time_match_scores if s >= 1.0) if time_channel_active else 0,
            'near_perfect_matches': sum(1 for s in time_match_scores if s >= 0.99) if time_channel_active else 0,
            'score_range': [min(time_match_scores), max(time_match_scores)] if time_match_scores else [0, 0]
        },
        'all_weighted_matches': {
            'min': float(min(all_weighted_matches)) if all_weighted_matches else 0,
            'max': float(max(all_weighted_matches)) if all_weighted_matches else 0,
            'mean': float(np.mean(all_weighted_matches)) if all_weighted_matches else 0,
            'std': float(np.std(all_weighted_matches)) if all_weighted_matches else 0,
            'count': len(all_weighted_matches)
        },
        'all_match_scores': {
            'min': float(min(all_match_scores)) if all_match_scores else 0,
            'max': float(max(all_match_scores)) if all_match_scores else 0,
            'mean': float(np.mean(all_match_scores)) if all_match_scores else 0,
            'std': float(np.std(all_match_scores)) if all_match_scores else 0,
            'count': len(all_match_scores)
        },
        'passing_weighted_matches': {
            'min': float(min([e['weighted_match'] for e in all_vigilant_events])) if all_vigilant_events else 0,
            'max': float(max([e['weighted_match'] for e in all_vigilant_events])) if all_vigilant_events else 0,
            'mean': float(np.mean([e['weighted_match'] for e in all_vigilant_events])) if all_vigilant_events else 0,
            'count': len(all_vigilant_events)
        }
    }

    print(f"Vigilance Stats: pass rate {vigilance_stats['vigilance_pass_rate']:.2%} "
          f"({vigilance_passed}/{vigilance_passed + vigilance_failed})")
    if time_channel_active:
        print(f"  Exact time matches: {exact_time_matches_found}")

    return {
        'top_k_events': top_k_events,
        'all_vigilant_events': all_vigilant_events if return_all_vigilant else [],
        'highest_scoring_event': highest_scoring_event,
        'vigilance_stats': vigilance_stats
    }


def get_time_sorted_events_normalized(retrieval_results, original_events):
    """Sort retrieved events by their normalized time values."""

    def process_event_list(event_list, sort_by_time=True):
        events_with_time = []

        for event_info in event_list:
            node_idx = event_info['node_id'] - 1
            if node_idx < len(original_events):
                event_data = original_events[node_idx].copy()

                event_data.update({
                    'retrieval_info': {
                        'node_id': event_info['node_id'],
                        'primary_score': event_info['primary_score'],
                        'match_scores': event_info['match_scores'],
                        'avg_match': event_info.get('avg_match', 0.0),
                        'weighted_match': event_info['weighted_match'],
                        'vigilance_passed': event_info['vigilance_passed'],
                        'is_empty_event': event_info.get('is_empty_event', False)
                    },
                    'normalized_time': event_info['normalized_time']
                })
                events_with_time.append(event_data)

        if sort_by_time:
            events_with_time.sort(key=lambda x: x.get('normalized_time', 0.0))

        return events_with_time

    result = {
        'top_k_events_time_sorted': process_event_list(retrieval_results['top_k_events'], sort_by_time=True),
        'top_k_events_match_sorted': process_event_list(retrieval_results['top_k_events'], sort_by_time=False),
        'all_vigilant_events_time_sorted': process_event_list(retrieval_results['all_vigilant_events'], sort_by_time=True),
        'all_vigilant_events_match_sorted': process_event_list(retrieval_results['all_vigilant_events'], sort_by_time=False),
        'highest_scoring_event': None,
        'vigilance_stats': retrieval_results['vigilance_stats']
    }

    if retrieval_results['highest_scoring_event']:
        result['highest_scoring_event'] = process_event_list(
            [retrieval_results['highest_scoring_event']], sort_by_time=False
        )[0]

    return result


def fill_empty_fields(events):
    """Fill empty fields with placeholders before processing."""
    earliest_timestamp = None

    for event in events:
        time_field = event.get("time")

        if not time_field or time_field == "" or time_field == 0:
            continue

        try:
            if isinstance(time_field, (int, float)):
                timestamp = float(time_field)
            elif isinstance(time_field, str):
                try:
                    dt = datetime.datetime.strptime(time_field, "%B %d, %Y")
                    timestamp = dt.timestamp()
                except ValueError:
                    try:
                        dt = datetime.datetime.strptime(time_field, "%Y-%m-%d")
                        timestamp = dt.timestamp()
                    except ValueError:
                        continue
            else:
                continue

            if earliest_timestamp is None or timestamp < earliest_timestamp:
                earliest_timestamp = timestamp

        except (ValueError, TypeError):
            continue

    if earliest_timestamp:
        earliest_datetime = datetime.datetime.fromtimestamp(earliest_timestamp)
        try:
            placeholder_datetime = earliest_datetime.replace(year=earliest_datetime.year - 1)
        except ValueError:
            placeholder_datetime = earliest_datetime.replace(year=earliest_datetime.year - 1, day=28)
        placeholder_time = placeholder_datetime.strftime("%B %d, %Y")
    else:
        placeholder_time = "January 1, 1900"

    print(f"Using placeholder time: {placeholder_time}")

    filled_events = []

    for event in events:
        filled_event = event.copy()

        if not filled_event.get("time") or filled_event["time"] == 0 or filled_event["time"] == "":
            filled_event["time"] = placeholder_time

        text_fields = ["spaces", "entities", "content"]
        for field in text_fields:
            if not filled_event.get(field) or filled_event[field] == "" or filled_event[field] is None:
                filled_event[field] = "none"

        filled_events.append(filled_event)

    return filled_events


def check_normalized_vectors_list(event_vector):
    time_list, spaces, entities, content = event_vector

    if not (len(time_list) == 1 and isinstance(time_list[0], (int, float))
            and 0 <= time_list[0] <= 1 and not math.isnan(time_list[0])):
        print(f"Invalid time value: {time_list}")
        return False

    def check_vec(vec, name):
        if vec is None:
            print(f"Missing vector: {name}")
            return False
        for i, v in enumerate(vec):
            if not (isinstance(v, (int, float)) and 0 <= v <= 1 and not math.isnan(v)):
                print(f"Invalid value in {name} at pos {i}: {v}")
                return False
        return True

    if not check_vec(spaces, "spaces"):
        return False
    if not check_vec(entities, "entities"):
        return False
    if not check_vec(content, "content"):
        return False

    return True


def extract_and_deduplicate_field(events, query_slots, retrieval_type):
    """Extract and deduplicate a specific field from retrieved events."""
    extracted_values = []

    retrieval_type_normalized = retrieval_type.lower()
    if retrieval_type_normalized in ['times']:
        retrieval_type_normalized = 'time'

    for event_data in events:
        extracted_value = None

        if retrieval_type_normalized == 'time':
            extracted_value = event_data.get("time")

        elif query_slots[0] != "*":
            if retrieval_type_normalized == "spaces":
                extracted_value = event_data.get("spaces")
            elif retrieval_type_normalized == "entities":
                extracted_value = event_data.get("entities")
            elif retrieval_type_normalized == "content":
                extracted_value = event_data.get("content")

        elif query_slots[1] != "*":
            if retrieval_type_normalized == "entities":
                extracted_value = event_data.get("entities")
            elif retrieval_type_normalized == "content":
                extracted_value = event_data.get("content")
            elif retrieval_type_normalized == "time":
                extracted_value = event_data.get("time")

        elif query_slots[2] != "*":
            if retrieval_type_normalized == "spaces":
                extracted_value = event_data.get("spaces")
            elif retrieval_type_normalized == "content":
                extracted_value = event_data.get("content")
            elif retrieval_type_normalized == "time":
                extracted_value = event_data.get("time")

        elif query_slots[3] != "*":
            if retrieval_type_normalized == "spaces":
                extracted_value = event_data.get("spaces")
            elif retrieval_type_normalized == "entities":
                extracted_value = event_data.get("entities")
            elif retrieval_type_normalized == "time":
                extracted_value = event_data.get("time")

        else:
            if retrieval_type_normalized == "spaces":
                extracted_value = event_data.get("spaces")
            elif retrieval_type_normalized == "entities":
                extracted_value = event_data.get("entities")
            elif retrieval_type_normalized == "content":
                extracted_value = event_data.get("content")
            elif retrieval_type_normalized == "time":
                extracted_value = event_data.get("time")

        if extracted_value is not None and extracted_value != "none":
            if isinstance(extracted_value, list):
                if len(extracted_value) > 0:
                    extracted_value_str = str(extracted_value[0])
                else:
                    continue
            else:
                extracted_value_str = str(extracted_value)

            if extracted_value_str.strip():
                extracted_values.append(extracted_value_str)

    seen = set()
    deduplicated = []
    for value in extracted_values:
        if value not in seen:
            seen.add(value)
            deduplicated.append(value)

    return deduplicated


def debug_time_normalization(vectorized_events, normalizer_stats):
    """Debug time normalization to identify issues."""
    print(f"\nTIME NORMALIZATION DEBUG")
    print("="*60)

    stored_times = []
    for i, event in enumerate(vectorized_events[:]):
        time_val = event[0][0] if event[0] and len(event[0]) > 0 else None
        stored_times.append(time_val)

    valid_times = [t for t in stored_times if t is not None]

    print(f"Sample stored normalized times (first 20 valid):")
    for i, time_val in enumerate(valid_times[:20]):
        print(f"  Event {i}: {time_val:.8f}")

    unique_times = set([f"{t:.8f}" for t in valid_times])

    print(f"\nTime distribution analysis:")
    print(f"  Total valid times: {len(valid_times)}")
    print(f"  Unique time values: {len(unique_times)}")
    print(f"  Min time: {min(valid_times):.8f}")
    print(f"  Max time: {max(valid_times):.8f}")
    print(f"  Range: {max(valid_times) - min(valid_times):.8f}")

    sorted_times = sorted(valid_times)
    time_gaps = [sorted_times[i+1] - sorted_times[i] for i in range(len(sorted_times)-1)]
    avg_gap = np.mean(time_gaps) if time_gaps else 0

    print(f"  Average gap between consecutive times: {avg_gap:.8f}")

    if len(unique_times) < 10:
        print(f"  WARNING: Very few unique times!")
        print(f"  All unique values: {sorted(unique_times)}")

    if avg_gap < 0.001:
        print(f"  WARNING: Very small gaps between times - may cause fuzzy matching issues")

    print(f"\nNormalization stats:")
    time_stats = normalizer_stats.get("time", {})
    print(f"  Stats available: {list(time_stats.keys())}")
    if "min" in time_stats and "max" in time_stats:
        print(f"  Min timestamp: {time_stats['min']}")
        print(f"  Max timestamp: {time_stats['max']}")
        print(f"  Timestamp range: {time_stats['max'] - time_stats['min']}")

    return {
        'unique_time_count': len(unique_times),
        'time_range': max(valid_times) - min(valid_times) if valid_times else 0,
        'average_gap': avg_gap,
        'needs_better_resolution': avg_gap < 0.001 or len(unique_times) < 10
    }


def run_event_retrieval_pipeline(qa_path, vectorized_events, stat_path, text_path,
                                  n=20, vigilance_threshold=[1.0, 0.99, 0.995, 0.98],
                                  max_retrievals=5,
                                  network_path="network_save_path"):
    """Run the event retrieval pipeline for a single book."""
    print(f"Loading QA from: {qa_path}")
    with open(qa_path) as f:
        qa_data = json.load(f)

    print(f"Loading text_events from: {text_path}")
    with open(text_path, "r") as f:
        text_events = json.load(f)

    print(f"Loading normalizer_stats from: {stat_path}")
    with open(stat_path, "r") as f:
        normalizer_stats = json.load(f)

    print(f"NORMALIZER STATS DEBUG:")
    print(f"  Available keys: {list(normalizer_stats.keys())}")
    if "time" in normalizer_stats:
        time_stats = normalizer_stats["time"]
        print(f"  Time stats keys: {list(time_stats.keys())}")
        print(f"  Time stats content: {time_stats}")
    else:
        print(f"  ERROR: No 'time' key found in normalizer stats!")
    print("="*60)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    fa = FusionART(numspace=4, lengths=[1, 384, 384, 384],
                   beta=[1.0]*4, alpha=[0.1]*4,
                   gamma=[1.0]*4, rho=[1.0]*4)
    ART2ACModelOverride(fa, k=1)
    ART2ACModelOverride(fa, k=2)
    ART2ACModelOverride(fa, k=3)

    def event_to_vector_list(event):
        return [
            [event["time"]],
            event["spaces"],
            event["entities"],
            event["content"]
        ]

    print("Encoding and storing events into FusionART")
    vectorized_events = [event_to_vector_list(e) for e in vectorized_events]

    debug_info = debug_time_normalization(vectorized_events, normalizer_stats)

    if debug_info['needs_better_resolution']:
        print(f"TIME NORMALIZATION ISSUE DETECTED!")
        print(f"  Issue: {debug_info}")

    for ve in vectorized_events:
        if check_normalized_vectors_list(ve):
            fa.setActivityF1(ve)
            J = fa.resSearch()
            fa.autoLearn(J)
        else:
            print(f"ERROR: Invalid event vector detected, setting to default for event: {ve}")
            fa.setActivityF1([[0.0], [0.0]*384, [0.0]*384, [0.0]*384])

    print("Saving FusionART network")
    saveFusionARTNetwork(fa, network_path)
    print("FusionART network saved successfully")

    print("Starting event retrieval")
    query_results = []

    for i, qa in enumerate(qa_data[:n]):
        print(f"\nQuery {i}: {qa.get('cue_completed')}")
        try:
            cue_completed = qa.get("cue_completed", "")
            cue_slots = [slot.strip() for slot in qa.get("cue", "(*, *, *, *)").strip("()").split(",")]
            cue_tokens = re.findall(r"{([^}]+)}", cue_completed)

            print(f"Cue slots: {cue_slots}")
            print(f"Cue tokens: {cue_tokens}")

            if not cue_tokens or len(cue_tokens) > 4:
                print(f"WARNING: Skipping Query {i}: Invalid cue format")
                continue

            query_vector = [[0.0], [0.0]*384, [0.0]*384, [0.0]*384]
            gamma = [0.0, 0.0, 0.0, 0.0]
            rho = vigilance_threshold

            slot_to_channel = {
                "t": 0, "T": 0, "time": 0, "times": 0, "Time": 0, "Times": 0,
                "l": 1, "L": 1, "loc": 1, "Loc": 1, "location": 1, "Location": 1,
                "locations": 1, "Locations": 1, "s": 1, "S": 1, "space": 1,
                "Space": 1, "spaces": 1, "Spaces": 1,
                "e": 2, "E": 2, "ent": 2, "Ent": 2, "entity": 2, "Entity": 2,
                "entities": 2, "Entities": 2,
                "c": 3, "C": 3, "con": 3, "Con": 3, "content": 3, "Content": 3,
                "*": None
            }

            token_idx = 0

            for idx, slot in enumerate(cue_slots):
                slot = slot.strip()

                if slot == "*":
                    continue

                if token_idx >= len(cue_tokens):
                    continue

                cue_text = cue_tokens[token_idx]
                token_idx += 1

                cleaned_slot = slot.lower().replace("{", "").replace("}", "").strip()

                channel_idx = slot_to_channel.get(cleaned_slot)

                if channel_idx is None:
                    channel_idx = idx
                    print(f"Using position-based mapping: slot {idx} -> channel {channel_idx}")

                print(f"Processing slot {idx}: '{slot}' -> cleaned: '{cleaned_slot}' -> channel {channel_idx}")

                if channel_idx is None:
                    print(f"WARNING: Skipping slot {idx}: could not determine channel")
                    continue

                if channel_idx == 0:
                    try:
                        ts = datetime.datetime.strptime(cue_text, "%B %d, %Y").timestamp()
                        min_v = normalizer_stats["time"]["min"]
                        max_v = normalizer_stats["time"]["max"]
                        norm_time = (ts - min_v) / (max_v - min_v if max_v != min_v else 1)
                        query_vector[channel_idx] = [norm_time]
                        gamma[channel_idx] = 1.0
                        rho[channel_idx] = vigilance_threshold[channel_idx]
                        print(f"Time channel activated: {cue_text} -> normalized {norm_time:.8f}")

                        if norm_time < 0 or norm_time > 1:
                            print(f"WARNING: Normalized time outside [0,1] range: {norm_time}")

                    except Exception as e:
                        print(f"ERROR: Time parse failed: {e}")
                        query_vector[0] = [0.0]
                        gamma[0] = 0.0
                        rho[0] = 0.0
                        continue

                elif channel_idx in [1, 2, 3]:
                    key_map = {1: "spaces", 2: "entities", 3: "content"}
                    key = key_map[channel_idx]

                    emb = model.encode(cue_text, normalize_embeddings=False)

                    emb_min, emb_max = emb.min(), emb.max()
                    if emb_max != emb_min:
                        normalized_embedding = ((emb - emb_min) / (emb_max - emb_min)).tolist()
                    else:
                        normalized_embedding = np.full_like(emb, 0.5).tolist()

                    query_vector[channel_idx] = normalized_embedding
                    gamma[channel_idx] = 1.0
                    rho[channel_idx] = vigilance_threshold[channel_idx]
                    print(f"{key.title()} channel activated: '{cue_text}' -> embedding shape {len(normalized_embedding)}")

            print(f"For this query, rho: {rho}, gamma: {gamma}")
            print(f"Final Query Configuration:")
            print(f"   Active channels: {[i for i, g in enumerate(gamma) if g > 0]}")
            print(f"   Gamma: {gamma}")
            for i, qv in enumerate(query_vector):
                if gamma[i] > 0:
                    if isinstance(qv, list) and len(qv) == 1:
                        print(f"   Channel {i}: {qv[0]:.8f} (scalar)")
                    else:
                        print(f"   Channel {i}: vector length {len(qv) if isinstance(qv, list) else 'unknown'}")

            fa_retrieve = FusionART(numspace=4, lengths=[1, 384, 384, 384],
                                    beta=[1.0]*4, alpha=[0.1]*4,
                                    gamma=gamma, rho=rho)

            loadFusionARTNetwork(fa_retrieve, network_path)
            ART2ACModelOverride(fa_retrieve, k=1)
            ART2ACModelOverride(fa_retrieve, k=2)
            ART2ACModelOverride(fa_retrieve, k=3)

            print(f"Using channel-specific vigilance: {vigilance_threshold}")

            raw_retrieval_results = retrieve_all_above_vigilance_with_exact_time_match(
                fa_retrieve, query_vector, gamma, vectorized_events,
                rho_threshold=vigilance_threshold,
                max_retrievals=max_retrievals,
                return_all_vigilant=True
            )

            print(f"Retrieval Summary:")
            print(f"   Top-k events: {len(raw_retrieval_results['top_k_events'])}")
            print(f"   All vigilant events: {len(raw_retrieval_results['all_vigilant_events'])}")

            if gamma[0] > 0:
                num_events = len(raw_retrieval_results['all_vigilant_events'])
                if num_events > 20:
                    print(f"WARNING: Time query returned {num_events} events - possible normalization issue")
                elif num_events == 0:
                    print(f"WARNING: No events found - vigilance threshold may be too strict")

            processed_events = get_time_sorted_events_normalized(raw_retrieval_results, text_events)

            def clean_event_data(event_data):
                return {
                    "time": event_data.get("time"),
                    "spaces": event_data.get("spaces"),
                    "entities": event_data.get("entities"),
                    "content": event_data.get("content"),
                    "match_score": event_data.get("retrieval_info", {}).get("primary_score", 0.0),
                    "weighted_match_score": event_data.get("retrieval_info", {}).get("weighted_match", 0.0),
                    "avg_match_score": event_data.get("retrieval_info", {}).get("avg_match", 0.0),
                    "individual_match_scores": event_data.get("retrieval_info", {}).get("match_scores", []),
                    "normalized_time": event_data.get("normalized_time", 0.0),
                    "vigilance_passed": event_data.get("retrieval_info", {}).get("vigilance_passed", False)
                }

            clean_results = {
                "top_k_events_time_sorted": [clean_event_data(e) for e in processed_events['top_k_events_time_sorted']],
                "top_k_events_match_sorted": [clean_event_data(e) for e in processed_events['top_k_events_match_sorted']],
                "all_vigilant_events_time_sorted": [clean_event_data(e) for e in processed_events['all_vigilant_events_time_sorted']],
                "all_vigilant_events_match_sorted": [clean_event_data(e) for e in processed_events['all_vigilant_events_match_sorted']],
                "highest_scoring_event": clean_event_data(processed_events['highest_scoring_event']) if processed_events['highest_scoring_event'] else None,
                "vigilance_stats": processed_events['vigilance_stats'],
                "extracted_field_values": {
                    "top_k_time_sorted": extract_and_deduplicate_field(
                        processed_events['top_k_events_time_sorted'], cue_slots, qa.get("retrieval_type", "Content")
                    ),
                    "top_k_match_sorted": extract_and_deduplicate_field(
                        processed_events['top_k_events_match_sorted'], cue_slots, qa.get("retrieval_type", "Content")
                    ),
                    "all_vigilant_time_sorted": extract_and_deduplicate_field(
                        processed_events['all_vigilant_events_time_sorted'], cue_slots, qa.get("retrieval_type", "Content")
                    ),
                    "all_vigilant_match_sorted": extract_and_deduplicate_field(
                        processed_events['all_vigilant_events_match_sorted'], cue_slots, qa.get("retrieval_type", "Content")
                    )
                }
            }

            query_results.append({
                "query_id": qa.get("q_idx", i),
                "query_text": cue_completed,
                "cue_slots": cue_slots,
                "cue_channel_names": [f"channel_{i}" for i in range(len(cue_slots))],
                "retrieval_type": qa.get("retrieval_type", "Content"),
                "results": clean_results,
                "num_top_k_retrieved": len(clean_results["top_k_events_time_sorted"]),
                "num_all_vigilant": len(clean_results["all_vigilant_events_time_sorted"]),
                "has_highest_scoring": clean_results["highest_scoring_event"] is not None,
                "vigilance_threshold": vigilance_threshold,
                "max_retrievals": max_retrievals,
                "retrieval_method": "match_based_with_exact_time"
            })

        except Exception as e:
            print(f"Error processing Query {i}: {e}")
            traceback.print_exc()
            continue

    return {
        "retrieval_results": query_results,
        "config": {
            "vigilance_threshold": vigilance_threshold,
            "max_retrievals": max_retrievals,
            "normalization_method": "per_vector_min_max",
            "time_sorting": "normalized_time_channel",
            "retrieval_method": "channel_specific_vigilance",
            "empty_fields_filled": True
        }
    }


def save_retrieval_results(results, output_path):
    """Save retrieval results to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=convert_np)

    print(f"Enhanced retrieval results saved to: {output_path}")

    retrieval_results = results.get("retrieval_results", [])
    total_queries = len(retrieval_results)

    queries_with_top_k = sum(1 for r in retrieval_results if r.get("num_top_k_retrieved", 0) > 0)
    queries_with_vigilant = sum(1 for r in retrieval_results if r.get("num_all_vigilant", 0) > 0)
    queries_with_highest = sum(1 for r in retrieval_results if r.get("has_highest_scoring", False))

    total_top_k_events = sum(r.get("num_top_k_retrieved", 0) for r in retrieval_results)
    total_vigilant_events = sum(r.get("num_all_vigilant", 0) for r in retrieval_results)

    total_nodes_checked = sum(r.get("results", {}).get("vigilance_stats", {}).get("total_nodes_checked", 0) for r in retrieval_results)
    total_vigilance_passed = sum(r.get("results", {}).get("vigilance_stats", {}).get("vigilance_passed", 0) for r in retrieval_results)
    total_vigilance_failed = sum(r.get("results", {}).get("vigilance_stats", {}).get("vigilance_failed", 0) for r in retrieval_results)

    overall_vigilance_pass_rate = total_vigilance_passed / (total_vigilance_passed + total_vigilance_failed) if (total_vigilance_passed + total_vigilance_failed) > 0 else 0

    print(f"Retrieval Summary:")
    print(f"   Total queries processed: {total_queries}")
    print(f"   Queries with top-k results: {queries_with_top_k} ({queries_with_top_k/total_queries:.1%})")
    print(f"   Queries with vigilant events: {queries_with_vigilant} ({queries_with_vigilant/total_queries:.1%})")
    print(f"   Total top-k events retrieved: {total_top_k_events}")
    print(f"   Total vigilant events found: {total_vigilant_events}")
    if total_queries > 0:
        print(f"   Average top-k events per query: {total_top_k_events/total_queries:.2f}")
        print(f"   Average vigilant events per query: {total_vigilant_events/total_queries:.2f}")
    print(f"   Overall vigilance pass rate: {overall_vigilance_pass_rate:.1%} ({total_vigilance_passed}/{total_nodes_checked} nodes)")


def run_retrieval_across_books(
    data_root: str = "data_root",
    n: int = 20,
    vigilance_threshold=[1.0, 0.99, 0.995, 0.98],
    max_retrievals: int = 5,
    network_path: str = "network_save_path"
):
    """Run event retrieval across multiple books."""
    book_folders = sorted([
        f for f in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, f)) and f.startswith("book")
    ])

    all_book_results = {}

    for book_name in tqdm(book_folders, desc="Processing books"):
        book_id = book_name.replace("book", "")
        book_path = os.path.join(data_root, book_name)

        qa_file = os.path.join(book_path, f"qa_book{book_id}.json")
        text_event_file = os.path.join(book_path, f"formatted_extracted_features_book{book_id}.json")
        vector_event_file = os.path.join(book_path, f"vectorized_features_book{book_id}.json")
        stat_file = os.path.join(book_path, "normalization_stats.json")

        if not os.path.exists(text_event_file):
            process_event_data(
                input_file=os.path.join(book_path, f"extracted_features_book{book_id}.json"),
                output_file=text_event_file
            )
            print(f"Processed text events for {book_name} and saved to {text_event_file}")
        else:
            print(f"Using existing text events for {book_name} from {text_event_file}")

        if not all(os.path.exists(f) for f in [qa_file, text_event_file]):
            print(f"WARNING: Missing required files for {book_name}, skipping...")
            continue

        with open(text_event_file, "r", encoding="utf-8") as f:
            text_events = json.load(f)

        text_events = fill_empty_fields(text_events)

        print(f"Existence of vectorized events: {os.path.exists(vector_event_file)}")
        print(f"Existence of normalization stats: {os.path.exists(stat_file)}")

        if not os.path.exists(vector_event_file) or not os.path.exists(stat_file):
            print(f"Re-normalizing events for {book_name} with per-vector min-max")
            normalizer = EventNormalizer(method="min_max_per_vector")
            vectorized_events = json.loads(json.dumps(text_events))

            normalizer.normalize_text_field(vectorized_events, "spaces")
            normalizer.normalize_text_field(vectorized_events, "entities")
            normalizer.normalize_text_field(vectorized_events, "content")
            normalizer.normalize_time_field(vectorized_events, "time")

            normalizer.save_stats(stat_file)
            with open(vector_event_file, "w", encoding="utf-8") as f:
                json.dump(vectorized_events, f, indent=2, default=convert_np)

        else:
            print(f"Using existing vectorized events and normalization stats for {book_name}")
            with open(vector_event_file, "r", encoding="utf-8") as f:
                vectorized_events = json.load(f)

        try:
            result = run_event_retrieval_pipeline(
                qa_file, vectorized_events, stat_file, text_event_file,
                n=n, vigilance_threshold=vigilance_threshold,
                max_retrievals=max_retrievals,
                network_path=network_path
            )

            all_book_results[book_id] = result

            retrieval_out = os.path.join(book_path, f"match_based_retrieval_results_book{book_id}.json")
            save_retrieval_results(result, retrieval_out)

            print(f"Retrieval complete for {book_name}")

        except Exception as e:
            print(f"ERROR processing {book_name}: {e}")
            traceback.print_exc()
            continue

    if all_book_results:
        combined_file = os.path.join(data_root, "all_books_match_based_retrieval_results.json")
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(all_book_results, f, indent=2, ensure_ascii=False, default=convert_np)

        print(f"All books processed successfully!")
        print(f"Combined results saved to: {combined_file}")
        print(f"Books processed: {len(all_book_results)}")
    else:
        print("ERROR: No books were successfully processed.")


def main():
    """Main function to run the event retrieval pipeline"""
    parser = argparse.ArgumentParser(description="Event Retrieval Pipeline")
    parser.add_argument("--data_root", default="data_root", help="Root dir with book folders")
    parser.add_argument("--limit", type=int, default=9999, help="QA pairs per book")
    parser.add_argument("--max_retrievals", type=int, default=20, help="Max events per query")
    parser.add_argument("--network_path", default="network_save_path", help="Path to save/load FusionART network")

    custom_rho = [1.0, 1.0, 1.0, 0.99]

    parser.add_argument("--time_vigilance", type=float, default=custom_rho[0], help="Time channel vigilance")
    parser.add_argument("--spaces_vigilance", type=float, default=custom_rho[1], help="Spaces channel vigilance")
    parser.add_argument("--entities_vigilance", type=float, default=custom_rho[2], help="Entities channel vigilance")
    parser.add_argument("--content_vigilance", type=float, default=custom_rho[3], help="Content channel vigilance")

    args = parser.parse_args()

    print(f"Using channel-specific vigilance:")
    print(f"   Time: {args.time_vigilance}")
    print(f"   Spaces: {args.spaces_vigilance}")
    print(f"   Entities: {args.entities_vigilance}")
    print(f"   Content: {args.content_vigilance}")

    run_retrieval_across_books(
        args.data_root,
        n=args.limit,
        vigilance_threshold=[args.time_vigilance, args.spaces_vigilance,
                              args.entities_vigilance, args.content_vigilance],
        max_retrievals=args.max_retrievals,
        network_path=args.network_path
    )


if __name__ == "__main__":
    main()
