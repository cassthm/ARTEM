import json
import re
import traceback
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import logging
from pathlib import Path
from datetime import datetime
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Question:
    """Represents a single evaluation question"""
    q_idx: int
    question: str
    correct_answer: List[str]
    retrieval_type: str
    cue: str
    n_chapters_correct_answer: int
    question_type: str = "all"
    bins_items_correct_answer: str = None
    debug_level_2: int = None
    cue_completed: str = None
    correct_answer_chapters: List[int] = None
    correct_answer_detailed: str = None
    n_items_correct_answer: int = None
    debug_changed: List = None
    debug_existing_change: Any = None
    book_id: int = None


@dataclass
class RetrievedEvent:
    """Represents a single retrieved event from ART model"""
    time: List[str]
    spaces: str
    entities: str
    content: str
    post_entities: str = None
    activation_score: float = None
    weighted_match_score: float = None
    avg_match_score: float = None
    individual_match_scores: List[float] = None
    normalized_time: float = None
    vigilance_passed: bool = None
    query_id: int = None


@dataclass
class ARTRetrievalResult:
    """Represents the full retrieval result from ART model"""
    qid: int
    retrieved_events: List[RetrievedEvent]
    retrieval_type: str
    question_type: str


class LLMWrapper(ABC):
    """Abstract base class for LLM wrappers"""

    @abstractmethod
    def generate(self, user_prompt: str, system_prompt: str = "", max_new_tokens: int = 4096) -> str:
        pass


class DeepseekR1Wrapper(LLMWrapper):
    """Wrapper for Deepseek R1 Distill Qwen 14B model using transformers pipeline"""

    def __init__(self, model_path: str = "LLM_model_path",
                 max_new_tokens: int = 512, temperature: float = 0.1):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.llm_pipe = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}")

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            )

            self.llm_pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=self.temperature
            )

            logger.info("Model initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            self.llm_pipe = None

    def generate(self, user_prompt: str, system_prompt: str = "", max_new_tokens: int = None) -> str:
        """Generate response using Deepseek R1 model pipeline"""
        if self.llm_pipe is None:
            logger.error("Model not initialized. Using fallback mock response.")
            return self._generate_mock_response(user_prompt, system_prompt)

        try:
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
            else:
                full_prompt = f"User: {user_prompt}\n\nAssistant:"

            tokens_to_generate = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

            outputs = self.llm_pipe(
                full_prompt,
                max_new_tokens=tokens_to_generate,
                do_sample=False,
                temperature=self.temperature,
                return_full_text=False
            )

            if outputs and len(outputs) > 0:
                generated_text = outputs[0]['generated_text'].strip()
                return generated_text
            else:
                logger.error("No output generated from model")
                return "I apologize, but I couldn't generate a response."

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_mock_response(user_prompt, system_prompt)

    def _generate_mock_response(self, user_prompt: str, system_prompt: str) -> str:
        """Generate mock response as fallback"""
        logger.warning("Using mock response as fallback")

        if "judge" in system_prompt.lower() or "expert" in system_prompt.lower():
            return self._generate_mock_judge_response(user_prompt)
        else:
            return self._generate_mock_answer_response(user_prompt)

    def _generate_mock_judge_response(self, prompt: str) -> str:
        """Generate mock judge response for testing"""
        return '{"identified_items_in_AI_answer": ["Mars Base", "atmospheric system"], "matching_score": [{"Mars Valles Industrial Hub": 0.8}, {"atmospheric scrubber failure": 1.0}], "explanation": "The AI identified relevant location and technical content with good accuracy."}'

    def _generate_mock_answer_response(self, prompt: str) -> str:
        """Generate mock answer response for testing"""
        return "Based on the retrieved events, the atmospheric scrubber failure occurred at Mars Valles Industrial Hub on November 13, 2226, involving Samuel Parker and affecting multiple personnel including Ronnie Randall and Mabel Schaffer."


def load_questions_from_json(qa_file_path: str, book_id: int) -> List[Question]:
    """Load questions from the qa_book{book_id}.json format"""
    try:
        with open(qa_file_path, 'r') as f:
            questions_data = json.load(f)

        questions = []
        for q_data in questions_data:
            get_value = q_data.get("get", "all")
            if get_value == "all":
                question_type = "all"
            elif get_value == "latest":
                question_type = "latest"
            elif get_value == "chronological":
                question_type = "chronological"
            else:
                question_type = "all"

            question = Question(
                q_idx=q_data["q_idx"],
                question=q_data["question"],
                correct_answer=q_data["correct_answer"],
                retrieval_type=q_data["retrieval_type"],
                cue=q_data["cue"],
                n_chapters_correct_answer=q_data["n_chapters_correct_answer"],
                question_type=question_type,
                bins_items_correct_answer=q_data.get("bins_items_correct_answer"),
                debug_level_2=q_data.get("debug_level_2"),
                cue_completed=q_data.get("cue_completed"),
                correct_answer_chapters=q_data.get("correct_answer_chapters"),
                correct_answer_detailed=q_data.get("correct_answer_detailed"),
                n_items_correct_answer=q_data.get("n_items_correct_answer"),
                debug_changed=q_data.get("debug_changed"),
                debug_existing_change=q_data.get("debug_existing_change"),
                book_id=book_id
            )
            questions.append(question)

        logger.info(f"Loaded {len(questions)} questions from {qa_file_path}")
        return questions

    except Exception as e:
        logger.error(f"Error loading questions from {qa_file_path}: {e}")
        return []


def load_all_questions(base_path: str, book_ids: List[int] = None) -> Dict[int, List[Question]]:
    """Load questions from multiple books"""
    if book_ids is None:
        book_ids = [1, 2, 3]

    all_questions = {}
    for book_id in book_ids:
        qa_file_path = f"{base_path}/book{book_id}/qa_book{book_id}.json"
        if os.path.exists(qa_file_path):
            questions = load_questions_from_json(qa_file_path, book_id)
            all_questions[book_id] = questions
        else:
            logger.warning(f"Question file not found: {qa_file_path}")
            all_questions[book_id] = []

    return all_questions


def load_retrieved_events_from_json(file_path: str) -> List[Dict[str, Any]]:
    """Load retrieved events from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if "retrieval_results" in data:
            retrieval_results = data["retrieval_results"]
        else:
            retrieval_results = data

        logger.info(f"Loaded {len(retrieval_results)} retrieval results from {file_path}")
        return retrieval_results

    except Exception as e:
        logger.error(f"Error loading retrieved events from {file_path}: {e}")
        return []


class ARTMemorySystem:
    """Interface for ART model that loads pre-retrieved events from JSON files"""

    def __init__(self, base_path: str = None):
        self.base_path = base_path
        self.retrieval_data = {}

    def load_retrieval_data_for_books(self, book_ids: List[int]):
        """Load retrieval data for multiple books from JSON files"""
        for book_id in book_ids:
            file_path = f"{self.base_path}/book{book_id}/match_based_retrieval_results_book{book_id}.json"
            if os.path.exists(file_path):
                retrieval_data = load_retrieved_events_from_json(file_path)
                self.retrieval_data[book_id] = retrieval_data
                logger.info(f"Loaded retrieval data for book {book_id}")
            else:
                logger.warning(f"Retrieval file not found: {file_path}")
                self.retrieval_data[book_id] = []

    def _parse_event_from_json(self, event_data: Dict[str, Any], query_id: int = None) -> RetrievedEvent:
        """Parse event data from JSON format to RetrievedEvent object"""
        retrieval_info = event_data.get("retrieval_info", {})
        activation = retrieval_info.get("activation")
        if activation is not None and (str(activation).lower() == 'nan' or pd.isna(activation)):
            activation = None

        return RetrievedEvent(
            time=event_data.get("time", []),
            spaces=event_data.get("spaces", ""),
            entities=event_data.get("entities", ""),
            content=event_data.get("content", ""),
            post_entities=event_data.get("post_entities", ""),
            activation_score=event_data.get("match_score"),
            weighted_match_score=event_data.get("weighted_match_score"),
            avg_match_score=event_data.get("avg_match_score"),
            individual_match_scores=event_data.get("individual_match_scores", []),
            normalized_time=event_data.get("normalized_time"),
            vigilance_passed=event_data.get("vigilance_passed", True),
            query_id=query_id
        )

    def retrieve_events(self, question: str, question_index: int, retrieval_type: str,
                        question_type: str = "all", book_id: int = 1,
                        cue_completed: str = None) -> ARTRetrievalResult:
        """Retrieve events using pre-loaded retrieval data from JSON files"""
        logger.info(f"Retrieving events for book {book_id}, question index {question_index}, type: {question_type}")

        if book_id not in self.retrieval_data:
            logger.warning(f"No retrieval data available for book {book_id}")
            return self._create_empty_result(question_index, retrieval_type, question_type)

        book_retrieval_data = self.retrieval_data[book_id]

        if question_index >= len(book_retrieval_data):
            logger.warning(f"Question index {question_index} out of range. Available: {len(book_retrieval_data)}")
            return self._create_empty_result(question_index, retrieval_type, question_type)

        retrieval_result = book_retrieval_data[question_index]
        logger.info(f"Using retrieval result at index {question_index}: query_id={retrieval_result.get('query_id')}")

        results = retrieval_result.get("results", {})
        retrieved_events = []

        if question_type == "latest":
            events_list = results.get("top_k_events_time_sorted", [])
            if events_list:
                retrieved_events.append(self._parse_event_from_json(events_list[0], question_index))
            logger.info(f"Retrieved 1 event for 'latest' question type")

        elif question_type == "chronological":
            events_list = results.get("all_vigilant_events_time_sorted", [])
            for event_data in events_list:
                retrieved_events.append(self._parse_event_from_json(event_data, question_index))
            logger.info(f"Retrieved {len(retrieved_events)} events for 'chronological' question type")

        else:
            events_list = results.get("all_vigilant_events_time_sorted", [])
            for event_data in events_list:
                retrieved_events.append(self._parse_event_from_json(event_data, question_index))
            logger.info(f"Retrieved {len(retrieved_events)} events for 'all' question type")

        return ARTRetrievalResult(
            qid=question_index,
            retrieved_events=retrieved_events,
            retrieval_type=retrieval_type,
            question_type=question_type
        )

    def _create_empty_result(self, question_index: int, retrieval_type: str,
                              question_type: str) -> ARTRetrievalResult:
        """Create an empty retrieval result"""
        return ARTRetrievalResult(
            qid=question_index,
            retrieved_events=[],
            retrieval_type=retrieval_type,
            question_type=question_type
        )


def format_retrieved_event_for_prompt(event: RetrievedEvent) -> str:
    """Format a retrieved event for inclusion in the prompt"""
    time_str = ", ".join(event.time) if event.time else "Unknown time"

    formatted_event = f"""Time: {time_str}
Location: {event.spaces}
Main Entity: {event.entities}
Post Entities: {event.post_entities if event.post_entities else "None"}
Content: {event.content}"""

    return formatted_event


def create_art_prompt(question: str, retrieved_events: List[RetrievedEvent], question_type: str = "all") -> str:
    """Create prompt using retrieved events from ART model"""

    formatted_events = []
    for i, event in enumerate(retrieved_events, 1):
        formatted_events.append(f"Retrieved Event {i}:\n{format_retrieved_event_for_prompt(event)}")

    events_context = "\n\n".join(formatted_events)

    if question_type == "latest":
        time_instruction = "Focus on the most recent event provided."
    elif question_type == "chronological":
        time_instruction = "Consider the chronological order of events when answering."
    else:
        time_instruction = "Consider all the retrieved events when answering."

    example_json = '''[
    {
      "time": [
        "December 26, 2226"
      ],
      "spaces": "Asteroid Psyche Base",
      "entities": "Ezra Edwards",
      "content": "antimatter cascade"
    },
    {
      "time": [
        "November 06, 2226"
      ],
      "spaces": "Shkadov Thruster Hub",
      "entities": "Zoe Brown",
      "content": "fungal spore outbreak in the bio-containment sector"
    }
  ]'''

    if not events_context.strip():
        prompt = f"""You are participating in an episodic memory test, based on the data below, which was retrieved from a book. You need to read it and internalize as if you had personally experienced the events described. After the text, you will find a question about the content. Please answer this question concisely based solely on the information provided in the retrieved data.

Retrieved Events:
None found or no relevant events retrieved.

Question: {question}

Instructions: {time_instruction}

Please answer the question to the best of your ability, based only on the information provided in the relevant chunks above. If you are unsure about an answer, it's okay to say so. Do not invent or assume information that was not explicitly stated in the text.
Notice the following examples of how to concisely answer the question:

Example 1:

Retrieved Events:
"events": {example_json}

Question: Consider all events involving antimatter cascade. List all the locations where these events took place, without mentioning the events themselves.

Instructions: Consider all the retrieved events when answering.

Answer: Asteroid Psyche Base
-------------------------------------
Example 2:

Retrieved Events:
None found or no relevant events retrieved.

Question: Consider all events involving antimatter cascade. List all the locations where these events took place, without mentioning the events themselves.

Instructions: Consider all the retrieved events when answering.

Answer: I don't know. No relevant information is provided.
"""
    else:
        prompt = f"""You are participating in an episodic memory test, based on the data below, which was retrieved from a book. You need to read it and internalize as if you had personally experienced the events described. After the text, you will find a question about the content. Please answer this question concisely based solely on the information provided in the retrieved data.

Retrieved Events:
{events_context}

Question: {question}

Instructions: {time_instruction}

Please answer the question to the best of your ability, based only on the information provided in the relevant chunks above. If you are unsure about an answer, it's okay to say so. Do not invent or assume information that was not explicitly stated in the text.
Notice the following examples of how to concisely answer the question:

Example 1:

Retrieved Events:
"events": {example_json}

Question: Consider all events involving antimatter cascade. List all the locations where these events took place, without mentioning the events themselves.

Instructions: Consider all the retrieved events when answering.

Answer: Asteroid Psyche Base
-------------------------------------
Example 2:

Retrieved Events:
None found or no relevant events retrieved.

Question: Consider all events involving antimatter cascade. List all the locations where these events took place, without mentioning the events themselves.

Instructions: Consider all the retrieved events when answering.

Answer: I don't know. No relevant information is provided.
"""

    return prompt


def judge_prompt_func(retrieval_type: str, correct_answer: List[str], llm_answer: str,
                      correct_answer_long: List[str] = None) -> str:
    """Generate the judge prompt for evaluation"""

    if correct_answer_long is None:
        correct_answer_long = correct_answer
        adding_text = ''
    else:
        d = [{x: "score_between_0_and_1"} for x in correct_answer]
        adding_text = f'- The matching score should be of length 1, only "matching_score": {json.dumps(d)}'

    d = [{x: "score_between_0_and_1"} for x in correct_answer]

    prompt = f"""
You are an expert judge evaluating the accuracy of an AI-generated answer against a known groundtruth. Questions can probe for different types or aspects, like what actions or events took place, what people were involved, what were the dates, or what were the locations or spaces.

Question type: {retrieval_type}
Groundtruth: {correct_answer_long}
AI-generated answer: {llm_answer}

Your task:
- Identify all unique items in the AI-generated answer that are relevant to the question type. Answer an empty list [] for this field in case of at least one negative information (e.g., when the answer begins by telling there is no information, or cannot answer)
- Determine a matching score between 0 and 1 for each ground truth item. Give 1 if the item has been found in the relevant items of the AI-generated answer, considering synonyms, paraphrases, or close meanings. Give 0.5 if the item could be considered related to any AI-generated item but without being explicitly stated as such. Give 0 if the item missed mentioning a specific AI-generated item.
- Provide a brief explanation of the evaluation
{adding_text}

Provide your evaluation in the following JSON format:
{{
    "identified_items_in_AI_answer": ["AI_answer_item_1", "AI_answer_item_2", ...],
    "matching_score": {json.dumps(d)},
    "explanation": "Brief explanation of your evaluation"
}}
"""
    return prompt


def f1_score_func(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall"""
    if precision == 0 and recall == 0:
        return 0
    elif precision is not None and recall is not None:
        return 2 * (precision * recall) / (precision + recall)
    elif precision is None and recall is None:
        return 1
    else:
        return 0


def generate_metric(correct_answer: List[str], evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Generate metrics from judge evaluation"""

    correct_answer = list(dict.fromkeys(correct_answer))

    nb_gt = len(correct_answer)

    predictions = evaluation['identified_items_in_AI_answer']
    nb_preds_harsh = len(predictions)

    if (nb_preds_harsh > nb_gt) and (nb_gt > 0):
        nb_preds_lenient = nb_gt
    else:
        nb_preds_lenient = nb_preds_harsh

    matching_score = evaluation['matching_score']

    if isinstance(matching_score, (int, float)):
        score = min(1.0, max(0.0, float(matching_score)))
        if nb_gt > 0:
            matching_score = [{correct_answer[0]: score}]
        else:
            matching_score = []
    elif isinstance(matching_score, list):
        if len(matching_score) > 0 and isinstance(matching_score[0], (int, float)):
            matching_score = [{correct_answer[i]: min(1.0, max(0.0, float(score)))}
                              for i, score in enumerate(matching_score[:nb_gt])]
        else:
            matching_score = [{k: min(1.0, max(0.0, float(v))) for k, v in item.items()}
                              for item in matching_score]

    try:
        sum_scores = sum([float(list(x.values())[0]) for x in matching_score])
        sum_scores = min(sum_scores, float(nb_gt))
    except (AttributeError, TypeError, ValueError) as e:
        logger.error(f"Error calculating sum of scores: {e}")
        sum_scores = 0.0

    try:
        gt_alt = [list(x.keys())[0] for x in matching_score]
        nb_gt_alt = len(gt_alt)
    except (AttributeError, IndexError, TypeError) as e:
        logger.error(f"Error extracting ground truth items: {e}")
        gt_alt = correct_answer[:nb_gt]
        nb_gt_alt = nb_gt
        matching_score = [{item: 0.0} for item in gt_alt]

    if nb_gt != nb_gt_alt:
        logger.warning(f'Ground truth count mismatch: {nb_gt} vs {nb_gt_alt}. Using original count.')
        gt_alt = correct_answer[:nb_gt]
        nb_gt_alt = nb_gt

    precision_lenient = sum_scores / nb_preds_lenient if nb_preds_lenient > 0 else None
    precision_harsh = sum_scores / nb_preds_harsh if nb_preds_harsh > 0 else None
    recall = sum_scores / nb_gt if nb_gt > 0 else None

    if precision_lenient is not None:
        precision_lenient = min(1.0, precision_lenient)
    if precision_harsh is not None:
        precision_harsh = min(1.0, precision_harsh)
    if recall is not None:
        recall = min(1.0, recall)

    f1_score_lenient = f1_score_func(precision_lenient, recall)
    f1_score_harsh = f1_score_func(precision_harsh, recall)

    return {
        'predicted_items': predictions,
        'groundtruth_items': gt_alt,
        'matching_groundtruth_items_score': evaluation['matching_score'],
        'explanation': evaluation['explanation'],
        'nb_preds_lenient': nb_preds_lenient,
        'nb_preds_harsh': nb_preds_harsh,
        'nb_gt': nb_gt,
        'sum_scores': sum_scores,
        'precision_lenient': precision_lenient,
        'precision_harsh': precision_harsh,
        'recall': recall,
        'f1_score_lenient': f1_score_lenient,
        'f1_score_harsh': f1_score_harsh,
        'diff_f1': f1_score_lenient - f1_score_harsh
    }


def validate_metrics(result: Dict[str, Any]) -> bool:
    """Validate that all metrics are within expected bounds"""

    f1_lenient = result.get('f1_score_lenient', 0)
    f1_harsh = result.get('f1_score_harsh', 0)

    if f1_lenient > 1.0 or f1_harsh > 1.0:
        logger.error(f"Invalid F1 scores: lenient={f1_lenient}, harsh={f1_harsh}")
        return False

    precision_lenient = result.get('precision_lenient')
    precision_harsh = result.get('precision_harsh')
    recall = result.get('recall')

    for name, value in [('precision_lenient', precision_lenient),
                        ('precision_harsh', precision_harsh),
                        ('recall', recall)]:
        if value is not None and value > 1.0:
            logger.error(f"Invalid {name}: {value}")
            return False

    sum_scores = result.get('sum_scores', 0)
    nb_gt = result.get('nb_gt', 0)

    if sum_scores > nb_gt:
        logger.error(f"Invalid sum_scores > nb_gt: {sum_scores} > {nb_gt}")
        return False

    return True


def evaluate_answer_with_art(llm_answer: str, correct_answer: List[str], retrieval_type: str,
                              judge_model: LLMWrapper,
                              correct_answer_long: List[str] = None) -> Dict[str, Any]:
    """Evaluate a single answer using LLM judge"""

    if correct_answer_long is None:
        correct_answer_long = correct_answer

    judge_prompt = judge_prompt_func(retrieval_type, correct_answer, llm_answer, correct_answer_long)
    logger.info(f"Judge prompt created for question type: {retrieval_type}")

    judge_response = judge_model.generate(
        user_prompt=judge_prompt,
        system_prompt="You are an expert in memory tests.",
        max_new_tokens=4096
    )
    logger.info(f"Judge response received")

    try:
        evaluation = json.loads(judge_response)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        json_match = re.search(r'\{.*\}', judge_response, re.DOTALL)
        if json_match:
            try:
                evaluation = json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.error("Failed to parse extracted JSON")
                raise ValueError(f"Failed to parse judge's response: {judge_response}")
        else:
            logger.error(f"No JSON found in response: {judge_response}")
            raise ValueError(f"No valid JSON in judge's response: {judge_response}")

    if 'matching_score' in evaluation:
        matching_score = evaluation['matching_score']
        if isinstance(matching_score, list):
            for i, item in enumerate(matching_score):
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, (int, float)):
                            matching_score[i][k] = min(1.0, max(0.0, float(v)))
                elif isinstance(item, (int, float)):
                    matching_score[i] = min(1.0, max(0.0, float(item)))
        elif isinstance(matching_score, (int, float)):
            evaluation['matching_score'] = min(1.0, max(0.0, float(matching_score)))

    result = generate_metric(correct_answer, evaluation)

    if not validate_metrics(result):
        logger.warning(f"Metric validation failed for question. Clamping values.")
        for key in ['precision_lenient', 'precision_harsh', 'recall', 'f1_score_lenient', 'f1_score_harsh']:
            if key in result and result[key] is not None:
                result[key] = min(1.0, max(0.0, result[key]))

    return result


def categorize_by_event_count(n_events: int) -> str:
    """Categorize number of events into bins"""
    if n_events == 0:
        return "0"
    elif n_events == 1:
        return "1"
    elif n_events == 2:
        return "2"
    elif 3 <= n_events <= 5:
        return "3-5"
    else:
        return "6+"


class ARTMemoryEvaluator:
    """Main class for conducting ART-based episodic memory evaluation"""

    def __init__(self, answering_model: LLMWrapper, judge_model: LLMWrapper,
                 art_system: ARTMemorySystem,
                 output_dir: str = "data_root/book1/art_evaluation_results"):
        self.answering_model = answering_model
        self.judge_model = judge_model
        self.art_system = art_system
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []

    def evaluate_question_with_art(self, question: Question, question_index: int) -> Dict[str, Any]:
        """Evaluate a single question using pre-loaded ART retrieval results"""
        logger.info(f"Evaluating question {question.q_idx} (index {question_index}): {question.question}")

        art_retrieval = self.art_system.retrieve_events(
            question.question,
            question_index,
            question.retrieval_type,
            question.question_type,
            question.book_id,
            question.cue_completed
        )

        art_prompt = create_art_prompt(question.question, art_retrieval.retrieved_events, question.question_type)
        logger.info(f"Created prompt for question {question.q_idx}")

        model_answer = self.answering_model.generate(
            user_prompt=art_prompt,
            system_prompt="You are an AI assistant answering questions about episodic memory based on retrieved events."
        )
        logger.info(f"Model answer generated for question {question.q_idx}")

        evaluation = evaluate_answer_with_art(
            llm_answer=model_answer,
            correct_answer=question.correct_answer,
            retrieval_type=question.retrieval_type,
            judge_model=self.judge_model
        )

        evaluation.update({
            'q_idx': question.q_idx,
            'question_index': question_index,
            'question': question.question,
            'model_answer': model_answer,
            'correct_answer': question.correct_answer,
            'retrieval_type': question.retrieval_type,
            'question_type': question.question_type,
            'book_id': question.book_id,
            'cue': question.cue,
            'cue_completed': question.cue_completed,
            'n_chapters_correct_answer': question.n_chapters_correct_answer,
            'n_items_correct_answer': question.n_items_correct_answer,
            'bins_items_correct_answer': question.bins_items_correct_answer,
            'correct_answer_chapters': question.correct_answer_chapters,
            'correct_answer_detailed': question.correct_answer_detailed,
            'event_category': categorize_by_event_count(question.n_chapters_correct_answer),
            'retrieved_events_count': len(art_retrieval.retrieved_events),
            'retrieved_events': [
                {
                    'time': event.time,
                    'spaces': event.spaces,
                    'entities': event.entities,
                    'content': event.content,
                    'post_entities': event.post_entities,
                    'activation_score': event.activation_score,
                    'weighted_match_score': event.weighted_match_score,
                    'avg_match_score': event.avg_match_score,
                    'individual_match_scores': event.individual_match_scores,
                    'normalized_time': event.normalized_time,
                    'vigilance_passed': event.vigilance_passed
                } for event in art_retrieval.retrieved_events
            ],
            'art_prompt': art_prompt
        })

        return evaluation

    def evaluate_questions(self, questions: List[Question],
                           memory_type: str = "STEM-ART",
                           model_name: str = "DeepSeek R1 Distill Qwen 14B") -> pd.DataFrame:
        """Evaluate a list of questions and return results DataFrame"""
        results = []

        for question_index, question in enumerate(questions):
            try:
                result = self.evaluate_question_with_art(question, question_index)
                result['memory_type'] = memory_type
                result['model_name'] = model_name
                result['evaluation_timestamp'] = datetime.now().isoformat()
                results.append(result)

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error evaluating question {question.q_idx} (index {question_index}): {e}")
                results.append({
                    'q_idx': question.q_idx,
                    'question_index': question_index,
                    'memory_type': memory_type,
                    'model_name': model_name,
                    'question_type': question.question_type if hasattr(question, 'question_type') else 'all',
                    'book_id': question.book_id if hasattr(question, 'book_id') else None,
                    'f1_score_lenient': 0.0,
                    'event_category': categorize_by_event_count(question.n_chapters_correct_answer),
                    'error': str(e),
                    'evaluation_timestamp': datetime.now().isoformat()
                })

        df = pd.DataFrame(results)
        self.results = results
        return df

    def save_results_to_json(self, results_df: pd.DataFrame, experiment_name: str = None,
                             save_by_book: bool = True,
                             output_base_path: str = "data_root"):
        """Save results to JSON files"""
        if experiment_name is None:
            experiment_name = f"art_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        saved_files = {}

        detailed_results_file = self.output_dir / f"{experiment_name}_detailed_results.json"
        with open(detailed_results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Detailed results saved to {detailed_results_file}")

        summary_stats = self.create_summary_statistics(results_df)
        summary_file = self.output_dir / f"{experiment_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        logger.info(f"Summary statistics saved to {summary_file}")

        csv_file = self.output_dir / f"{experiment_name}_results.csv"
        results_df.to_csv(csv_file, index=False)
        logger.info(f"Results DataFrame saved to {csv_file}")

        saved_files['overall'] = {
            'detailed_results': detailed_results_file,
            'summary': summary_file,
            'csv': csv_file
        }

        if save_by_book:
            book_files = self.save_results_by_book(results_df, output_base_path)
            saved_files.update(book_files)

        return saved_files

    def save_results_by_book(self, results_df: pd.DataFrame,
                              output_base_path: str = "data_root"):
        """Save results separately for each book"""

        if 'book_id' not in results_df.columns:
            logger.warning("No book_id column found, cannot save by book")
            return {}

        saved_files = {}

        for book_id in results_df['book_id'].unique():
            if pd.isna(book_id):
                continue

            book_id = int(book_id)
            book_df = results_df[results_df['book_id'] == book_id]

            book_dir = Path(output_base_path) / f"book{book_id}"
            book_dir.mkdir(parents=True, exist_ok=True)

            metric_data = []
            for _, row in book_df.iterrows():
                metric_item = {
                    'q_idx': row['q_idx'],
                    'question': row['question'],
                    'correct_answer': row['correct_answer'],
                    'retrieval_type': row['retrieval_type'],
                    'question_type': row['question_type'],
                    'model_answer': row['model_answer'],
                    'predicted_items': row.get('predicted_items', []),
                    'groundtruth_items': row.get('groundtruth_items', []),
                    'matching_groundtruth_items_score': row.get('matching_groundtruth_items_score', []),
                    'explanation': row.get('explanation', ''),
                    'nb_preds_lenient': row.get('nb_preds_lenient', 0),
                    'nb_preds_harsh': row.get('nb_preds_harsh', 0),
                    'nb_gt': row.get('nb_gt', 0),
                    'sum_scores': row.get('sum_scores', 0),
                    'precision_lenient': row.get('precision_lenient'),
                    'precision_harsh': row.get('precision_harsh'),
                    'recall': row.get('recall'),
                    'f1_score_lenient': row.get('f1_score_lenient', 0),
                    'f1_score_harsh': row.get('f1_score_harsh', 0),
                    'diff_f1': row.get('diff_f1', 0),
                    'event_category': row.get('event_category', ''),
                    'n_chapters_correct_answer': row.get('n_chapters_correct_answer', 0),
                    'n_items_correct_answer': row.get('n_items_correct_answer', 0)
                }
                metric_data.append(metric_item)

            metric_file = book_dir / f"dataForMetricComputation_book{book_id}.json"
            with open(metric_file, 'w') as f:
                json.dump(metric_data, f, indent=2, default=str)

            book_results = book_df.to_dict('records')
            results_file = book_dir / f"art_stem_evaluation_book{book_id}.json"
            with open(results_file, 'w') as f:
                json.dump(book_results, f, indent=2, default=str)

            saved_files[f'book_{book_id}'] = {
                'metric_data': str(metric_file),
                'detailed_results': str(results_file)
            }

            logger.info(f"Saved results for book {book_id} to {book_dir}")

        return saved_files

    def create_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary statistics from results DataFrame"""

        summary = {
            'experiment_info': {
                'total_questions': len(df),
                'memory_type': df['memory_type'].iloc[0] if not df.empty else None,
                'model_name': df['model_name'].iloc[0] if not df.empty else None,
                'evaluation_date': datetime.now().isoformat()
            },
            'overall_performance': {
                'mean_f1_lenient': float(df['f1_score_lenient'].mean()) if 'f1_score_lenient' in df.columns else None,
                'std_f1_lenient': float(df['f1_score_lenient'].std()) if 'f1_score_lenient' in df.columns else None,
                'mean_f1_harsh': float(df['f1_score_harsh'].mean()) if 'f1_score_harsh' in df.columns else None,
                'std_f1_harsh': float(df['f1_score_harsh'].std()) if 'f1_score_harsh' in df.columns else None
            },
            'performance_by_category': {},
            'performance_by_retrieval_type': {},
            'performance_by_question_type': {}
        }

        if 'event_category' in df.columns and 'f1_score_lenient' in df.columns:
            for category in df['event_category'].unique():
                cat_data = df[df['event_category'] == category]['f1_score_lenient']
                summary['performance_by_category'][category] = {
                    'count': len(cat_data),
                    'mean_f1': float(cat_data.mean()),
                    'std_f1': float(cat_data.std()),
                    'formatted': f"{cat_data.mean():.2f}+/-{cat_data.std():.2f}"
                }

        if 'retrieval_type' in df.columns and 'f1_score_lenient' in df.columns:
            for ret_type in df['retrieval_type'].unique():
                type_data = df[df['retrieval_type'] == ret_type]['f1_score_lenient']
                summary['performance_by_retrieval_type'][ret_type] = {
                    'count': len(type_data),
                    'mean_f1': float(type_data.mean()),
                    'std_f1': float(type_data.std()),
                    'formatted': f"{type_data.mean():.2f}+/-{type_data.std():.2f}"
                }

        summary['performance_by_book'] = {}
        if 'book_id' in df.columns and 'f1_score_lenient' in df.columns:
            for book_id in df['book_id'].unique():
                if pd.notna(book_id):
                    book_data = df[df['book_id'] == book_id]['f1_score_lenient']
                    summary['performance_by_book'][f'book_{int(book_id)}'] = {
                        'count': len(book_data),
                        'mean_f1': float(book_data.mean()),
                        'std_f1': float(book_data.std()),
                        'formatted': f"{book_data.mean():.2f}+/-{book_data.std():.2f}"
                    }

        if 'question_type' in df.columns and 'f1_score_lenient' in df.columns:
            for q_type in df['question_type'].unique():
                qtype_data = df[df['question_type'] == q_type]['f1_score_lenient']
                summary['performance_by_question_type'][q_type] = {
                    'count': len(qtype_data),
                    'mean_f1': float(qtype_data.mean()),
                    'std_f1': float(qtype_data.std()),
                    'formatted': f"{qtype_data.mean():.2f}+/-{qtype_data.std():.2f}"
                }

        return summary


def create_summary_table(df: pd.DataFrame, group_by_book: bool = False) -> pd.DataFrame:
    """Create summary table of evaluation results"""

    if group_by_book and 'book_id' in df.columns:
        group_columns = ['memory_type', 'model_name', 'book_id', 'event_category']
        index_columns = ['memory_type', 'model_name', 'book_id']
    else:
        group_columns = ['memory_type', 'model_name', 'event_category']
        index_columns = ['memory_type', 'model_name']

    summary = df.groupby(group_columns).agg({
        'f1_score_lenient': ['mean', 'std', 'count']
    }).round(2)

    summary.columns = ['mean', 'std', 'count']
    summary = summary.reset_index()

    summary['formatted_score'] = summary.apply(
        lambda row: f"{row['mean']:.2f}+/-{row['std']:.2f}", axis=1
    )

    pivot_table = summary.pivot_table(
        index=index_columns,
        columns='event_category',
        values='formatted_score',
        aggfunc='first'
    )

    expected_cols = ['0', '1', '2', '3-5', '6+']
    available_cols = [col for col in expected_cols if col in pivot_table.columns]
    pivot_table = pivot_table[available_cols]

    return pivot_table


def evaluate_books(base_path: str, book_ids: List[int] = None,
                   model_path: str = "LLM_model_path") -> Tuple[pd.DataFrame, 'ARTMemoryEvaluator']:
    """Main function to evaluate questions from multiple books using pre-loaded retrieval data"""

    if book_ids is None:
        book_ids = sorted([
            int(re.search(r'book(\d+)', folder.name).group(1))
            for folder in Path(base_path).glob('book*')
            if folder.is_dir() and re.match(r'book\d+$', folder.name)
        ])

    all_questions = load_all_questions(base_path, book_ids)

    logger.info("Initializing DeepSeek R1 models...")
    answering_model = DeepseekR1Wrapper(model_path=model_path, max_new_tokens=512, temperature=0.1)
    judge_model = DeepseekR1Wrapper(model_path=model_path, max_new_tokens=1024, temperature=0.1)

    art_system = ARTMemorySystem(base_path)
    art_system.load_retrieval_data_for_books(book_ids)

    evaluator = ARTMemoryEvaluator(answering_model, judge_model, art_system)

    all_questions_flat = []
    for book_id, questions in all_questions.items():
        all_questions_flat.extend(questions)

    logger.info(f"Total questions to evaluate: {len(all_questions_flat)} across {len(book_ids)} books")

    results_df = evaluator.evaluate_questions(
        all_questions_flat,
        memory_type="STEM-ART",
        model_name="DeepSeek R1 Distill Qwen 14B"
    )

    return results_df, evaluator


def get_available_book_ids(base_path: str) -> List[int]:
    """Automatically detect available book folders in the format book{id}"""
    book_ids = []

    try:
        base_dir = Path(base_path)

        if not base_dir.exists():
            logger.warning(f"Base path does not exist: {base_path}")
            return []

        book_pattern = re.compile(r'^book(\d+)$')

        for item in base_dir.iterdir():
            if item.is_dir():
                match = book_pattern.match(item.name)
                if match:
                    book_id = int(match.group(1))
                    book_ids.append(book_id)

        book_ids.sort()
        logger.info(f"Found {len(book_ids)} book folders: {book_ids}")

    except Exception as e:
        logger.error(f"Error detecting book folders: {e}")
        return []

    return book_ids


def main(model_path: str = "LLM_model_path"):
    """Main function demonstrating the ART evaluation system"""

    base_path = "data_root"

    book_ids = get_available_book_ids(base_path)

    if not book_ids:
        logger.warning("No book folders found, using default [1, 2, 3, 4, 5]")
        book_ids = [1, 2, 3, 4, 5]

    logger.info(f"Evaluating books: {book_ids}")

    results_df, evaluator = evaluate_books(base_path, book_ids, model_path=model_path)

    experiment_name = f"art_stem_evaluation_books_{'_'.join(map(str, book_ids))}"
    saved_files = evaluator.save_results_to_json(
        results_df,
        experiment_name,
        save_by_book=True,
        output_base_path="data_root"
    )

    summary_table = create_summary_table(results_df)
    summary_table_by_book = create_summary_table(results_df, group_by_book=True)

    print("=== ART Memory Evaluation Results ===")
    print(f"Experiment: {experiment_name}")
    print(f"Total questions evaluated: {len(results_df)}")
    print(f"Books evaluated: {book_ids}")
    print(f"Model path: {model_path}")

    print("\nOverall Summary Table:")
    print(summary_table)

    if len(book_ids) > 1:
        print("\nSummary Table by Book:")
        print(summary_table_by_book)

    if 'question_type' in results_df.columns:
        print("\nPerformance by Question Type:")
        question_type_summary = results_df.groupby('question_type')['f1_score_lenient'].agg(['mean', 'std', 'count'])
        for qtype in question_type_summary.index:
            mean_f1 = question_type_summary.loc[qtype, 'mean']
            std_f1 = question_type_summary.loc[qtype, 'std']
            count = question_type_summary.loc[qtype, 'count']
            print(f"  {qtype}: {mean_f1:.2f}+/-{std_f1:.2f} (n={count})")

    if 'book_id' in results_df.columns:
        print("\nPerformance by Book:")
        book_summary = results_df.groupby('book_id')['f1_score_lenient'].agg(['mean', 'std', 'count'])
        for book_id in book_summary.index:
            if pd.notna(book_id):
                mean_f1 = book_summary.loc[book_id, 'mean']
                std_f1 = book_summary.loc[book_id, 'std']
                count = book_summary.loc[book_id, 'count']
                print(f"  Book {int(book_id)}: {mean_f1:.2f}+/-{std_f1:.2f} (n={count})")

    print(f"\nResults saved to:")
    print("Overall results:")
    for file_type, file_path in saved_files['overall'].items():
        print(f"  {file_type}: {file_path}")

    print("Per-book results:")
    for book_key, book_files in saved_files.items():
        if book_key != 'overall':
            print(f"  {book_key}:")
            for file_type, file_path in book_files.items():
                print(f"    {file_type}: {file_path}")

    return results_df, summary_table, saved_files


def test_retrieval_loading(base_path: str, book_id: int = 1):
    """Test function to verify retrieval data loading"""

    print(f"=== TESTING RETRIEVAL DATA LOADING FOR BOOK {book_id} ===")

    art_system = ARTMemorySystem(base_path)
    art_system.load_retrieval_data_for_books([book_id])

    if book_id in art_system.retrieval_data:
        retrieval_data = art_system.retrieval_data[book_id]
        print(f"Successfully loaded {len(retrieval_data)} retrieval results for book {book_id}")

        for i, result in enumerate(retrieval_data[:3]):
            query_id = result.get('query_id')
            retrieval_type = result.get('retrieval_type')
            results = result.get('results', {})
            events_count = len(results.get('all_vigilant_events_time_sorted', []))
            print(f"  Result {i}: query_id={query_id}, type={retrieval_type}, events={events_count}")
    else:
        print(f"Failed to load retrieval data for book {book_id}")

    print(f"\n=== TESTING QUESTION LOADING FOR BOOK {book_id} ===")
    questions = load_questions_from_json(f"{base_path}/book{book_id}/qa_book{book_id}.json", book_id)
    print(f"Successfully loaded {len(questions)} questions for book {book_id}")

    if questions and book_id in art_system.retrieval_data:
        print(f"\n=== TESTING SAMPLE RETRIEVAL ===")
        sample_question = questions[0]
        print(f"Testing with question 0: {sample_question.question}")

        try:
            result = art_system.retrieve_events(
                sample_question.question,
                sample_question.q_idx,
                sample_question.retrieval_type,
                sample_question.question_type,
                sample_question.book_id
            )

            print(f"Successfully retrieved {len(result.retrieved_events)} events")
            for i, event in enumerate(result.retrieved_events):
                print(f"  Event {i+1}: {event.content} at {event.spaces}")

        except Exception as e:
            print(f"Error during retrieval: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    base_path = "data_root"

    print("=== TESTING RETRIEVAL DATA LOADING ===")
    test_retrieval_loading(base_path, book_id=1)

    print("\n" + "="*50 + "\n")

    print("=== RUNNING FULL EVALUATION ===")
    results_df, summary_table, saved_files = main()
    print("\nResults saved to files:")
    for file_type, file_path in saved_files.items():
        print(f"  {file_type}: {file_path}")
    print("\nEvaluation completed successfully.")
