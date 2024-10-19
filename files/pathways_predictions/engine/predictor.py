#!/usr/bin/python3
# ECE 498B 2022
# Ben Chapman-Kish

# Prediction engine which converts data into numpy arrays for input to machine learning models,
# and uses a model selector to always get suggestions from the best-performing model so far.

from __future__ import annotations
from shared.types import *
from engine.models import BaseModel, JOB_MODELS, COURSE_MODELS
from engine.selector import ModelSelector

from timeit import default_timer

class PredictionEngine(object):
    def __init__(self, job_models: list[BaseModel] = JOB_MODELS, course_models: list[BaseModel] = COURSE_MODELS):
        self.postings_cache: dict[PostingType, dict[UUID, list[float]]] = {
            PostingType.JOB: {},
            PostingType.COURSE: {}
        }

        self.x_data_cache: dict[PostingType, list[list[float]]] = {
            PostingType.JOB: [],
            PostingType.COURSE: []
        }

        self.y_data_cache: dict[PostingType, list[float]] = {
            PostingType.JOB: [],
            PostingType.COURSE: []
        }

        self.reviews_cache: set[UUID] = set()

        self.trained_review_count = {
            PostingType.JOB: 0,
            PostingType.COURSE: 0
        }

        self.new_cached_postings = 0
        self.removed_cached_postings = 0
        self.new_cached_reviews = 0

        self.model_selector = ModelSelector(job_models, course_models)
    
    def cache_postings(self, postings: list[Posting]) -> None:
        for posting in postings:
            if len(posting.features) == 0:
                del self.postings_cache[posting.type][posting.id]
                self.removed_cached_postings += 1
                continue

            if posting.id not in self.postings_cache[posting.type]:
                self.postings_cache[posting.type][posting.id] = posting.features
                self.new_cached_postings += 1
    
    def get_new_cached_postings_and_reset(self) -> tuple[int, int]:
        new_postings = self.new_cached_postings
        removed_postings = self.removed_cached_postings

        self.new_cached_postings = 0
        self.removed_cached_postings = 0

        return (new_postings, removed_postings)
    
    def cache_reviews(self, reviews: list[Review]) -> None:
        for review in reviews:
            if review.id not in self.reviews_cache:
                self.x_data_cache[review.posting_type].append(review.features)
                self.y_data_cache[review.posting_type].append(review.rating)
                self.reviews_cache.add(review.id)

                self.new_cached_reviews += 1
    
    def get_new_cached_reviews_and_reset(self) -> int:
        new_reviews = self.new_cached_reviews

        self.new_cached_reviews = 0

        return new_reviews

    def learn_new_reviews(self) -> None:
        for posting_type in (PostingType.JOB, PostingType.COURSE):
            self.learn_new_reviews_of_type(posting_type)

    def learn_new_reviews_of_type(self, posting_type: PostingType) -> None:
        trained_reviews = self.trained_review_count[posting_type]
        untrained_reviews = len(self.y_data_cache[posting_type]) - trained_reviews

        if untrained_reviews > Config.data_buffer_spill_threshold:
            log.debug(f"Buffer for {posting_type.name} reviews has now reached spill quota, fitting models to {untrained_reviews} reviews")

            x_data = np.array(self.x_data_cache[posting_type][trained_reviews:])
            y_data = np.array(self.y_data_cache[posting_type][trained_reviews:])
            self.model_selector.learn(posting_type, x_data, y_data)

            self.trained_review_count[posting_type] += untrained_reviews

    def can_make_suggestions(self, posting_type: PostingType) -> bool:
        return self.trained_review_count[posting_type] >= Config.minimum_training_data_count
    
    @staticmethod
    def evaluate_filter(features: list[float], filter: PostingFilter) -> bool:
        a = features[filter.field_index]
        b = filter.target
        
        if filter.comparator == FilterComparator.EQUAL:
            return a == b
        elif filter.comparator == FilterComparator.NOT_EQUAL:
            return a != b
        elif filter.comparator == FilterComparator.GREATER_THAN:
            return a > b
        elif filter.comparator == FilterComparator.LESS_THAN:
            return a < b
        
        return False
    
    def _get_postings_with_filter(self, posting_type: PostingType, filters: list[PostingFilter]) -> list[tuple[UUID, list[float]]]:
        # O(p) cache dump, could be more efficient (pre-calculate the list)
        postings = self.postings_cache[posting_type].items()

        if len(filters) == 0:
            return list(postings)

        filtered_postings: list[tuple[UUID, list[float]]] = []

        for id, features in postings:
            # O(p*f) traversal over every pair of filters for each posting
            if all((self.evaluate_filter(features, filter) for filter in filters)): # O(1) integer comparison per filter
                filtered_postings.append((id, features)) # Overhead of tuple creation, list appending

        return filtered_postings
    
    # The most time-critical part of the system, be very thorough here:
    def make_suggestions_for_user(self, user_features: list[float], posting_type: PostingType) -> Optional[list[UUID]]:
        assert self.can_make_suggestions(posting_type) # O(1) integer comparison
        
        timing_start = default_timer()

        #postings = self._get_postings_with_filter(posting_type, filters) # O(p*f) complex data preparation

        # O(p) cache dump, could be more efficient (pre-calculate the list)
        postings = list(self.postings_cache[posting_type].items())
        
        timing_end = default_timer()
        elapsed = timing_end - timing_start
        log.info(f"[Suggestion Generation Timing] Posting retrieval time: {elapsed*1000:.0f}ms")


        if len(postings) < 1: # O(1) size check
            return None

        timing_start = default_timer()

        x_data = np.array([user_features + p_features for _, p_features in postings]) # O(p) iteration over postings, O(1) array creation

        timing_end = default_timer()
        elapsed = timing_end - timing_start
        log.info(f"[Suggestion Generation Timing] Data concatenation time: {elapsed*1000:.0f}ms")

        results = self.model_selector.predict(posting_type, x_data) # O(?) many things happen here
        log.info(f"Generated {len(results)} valid suggestions, extracting the top {Config.top_predictions_limit}")

        timing_start = default_timer()

        if len(results) < 1: # O(1) size check
            return None

        top_results = results[:-Config.top_predictions_limit-1:-1] # O(1) fixed-size reverse-order list slice
        results_ids = [postings[index][0] for index, _ in top_results] # O(1) fixed-size index lookup and attribute extraction

        timing_end = default_timer()
        elapsed = timing_end - timing_start
        log.info(f"[Suggestion Generation Timing] Data filtering time: {elapsed*1000:.0f}ms")

        return results_ids
