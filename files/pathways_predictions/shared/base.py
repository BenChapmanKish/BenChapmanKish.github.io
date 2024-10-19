#!/usr/bin/python3
# ECE 498B 2022
# Ben Chapman-Kish

# Shared imports and globals

from __future__ import annotations
import logging

class Config(object):
    _CONFIG_INSTANTIATED = False

    LOCALHOST = 'localhost'
    ANY_INTERFACE = '0.0.0.0'
    DEFAULT_ENGINE_PORT = 65071
    DEFAULT_SSH_PORT = 22
    INITIAL_LOCAL_PORT = 61001

    _DEFAULT_PING_TIMEOUT = 5.0
    _DEFAULT_REQUEST_TIMEOUT = 120.0
    _DEFAULT_REQUEST_BATCH_SIZE = 100
    _DEFAULT_BACKEND_SERVER_POLL_INTERVAL_MINUTES = 10
    _DEFAULT_TOP_PREDICTIONS_LIMIT = 10
    _DEFAULT_BUFFER_SPILL_THRESHOLD = 100
    _DEFAULT_MINIMUM_TRAINING_COUNT = 500
    _DEFAULT_VALIDATION_SPLIT_KFOLDS = 5
    _DEFAULT_CATEGORICAL_LABEL_SMOOTHING = 0.1
    _DEFAULT_FAKE_RATINGS_RANGE = (2,11)
    _DEFAULT_FAKE_RATINGS_PROBABILITY = 0.7
    _log_formatter = logging.Formatter("%(asctime)s %(levelname)s [%(module)s:%(lineno)d] %(message)s")

    @classmethod
    def instantiate(cls,
        debug_server: bool = False,
        verbose_training: bool = False,
        engine_host: str = ANY_INTERFACE,
        engine_port: int = DEFAULT_ENGINE_PORT,
        api_host: str | None = ANY_INTERFACE,
        api_port: int | None = None,

        engine_server_addrs: list[Config.ServerAddr] = [],

        ping_timeout: float = _DEFAULT_PING_TIMEOUT,
        request_timeout: float = _DEFAULT_REQUEST_TIMEOUT,
        request_payload_batch_size: int = _DEFAULT_REQUEST_BATCH_SIZE,
        backend_server_poll_interval_minutes: int = _DEFAULT_BACKEND_SERVER_POLL_INTERVAL_MINUTES,
        top_predictions_limit: int = _DEFAULT_TOP_PREDICTIONS_LIMIT,
        data_buffer_spill_threshold: int = _DEFAULT_BUFFER_SPILL_THRESHOLD,
        minimum_training_data_count: int = _DEFAULT_MINIMUM_TRAINING_COUNT,
        validation_split_kfolds: int = _DEFAULT_VALIDATION_SPLIT_KFOLDS,
        categorical_label_smoothing: float = _DEFAULT_CATEGORICAL_LABEL_SMOOTHING,
        fake_job_ratings_per_user_range: tuple[int, int] = _DEFAULT_FAKE_RATINGS_RANGE,
        fake_course_ratings_per_user_range: tuple[int, int] = _DEFAULT_FAKE_RATINGS_RANGE,
        fake_ratings_per_user_probability: float = _DEFAULT_FAKE_RATINGS_PROBABILITY,

        log_level: int = logging.NOTSET
    ):
        assert(cls._CONFIG_INSTANTIATED == False)
        cls._CONFIG_INSTANTIATED = True

        cls.debug_server = debug_server
        cls.verbose_training = verbose_training
        cls.api_host = api_host
        cls.api_port = api_port
        cls.engine_host = engine_host
        cls.engine_port = engine_port
        cls.engine_server_addrs = engine_server_addrs or [Config.ServerAddr(cls.LOCALHOST, cls.DEFAULT_ENGINE_PORT)]
        cls.ping_timeout = ping_timeout
        cls.request_timeout = request_timeout
        cls.request_payload_batch_size = request_payload_batch_size
        cls.backend_server_poll_interval_minutes = backend_server_poll_interval_minutes

        cls.top_predictions_limit = top_predictions_limit
        cls.data_buffer_spill_threshold = data_buffer_spill_threshold
        cls.minimum_training_data_count = minimum_training_data_count
        cls.validation_split_kfolds = validation_split_kfolds
        cls.categorical_label_smoothing = categorical_label_smoothing
        cls.fake_job_ratings_per_user_range = fake_job_ratings_per_user_range
        cls.fake_course_ratings_per_user_range = fake_course_ratings_per_user_range
        cls.fake_ratings_per_user_probability = fake_ratings_per_user_probability

        cls.set_log_level(log_level)
    
    @classmethod
    def set_log_level(cls, log_level: int) -> None:
        if log is None or log_level <= logging.NOTSET: return

        (log.removeHandler(handler) for handler in log.handlers)
        ch = logging.StreamHandler()
        ch.setFormatter(cls._log_formatter)
        ch.setLevel(log_level)
        log.setLevel(log_level)
        log.addHandler(ch)
    
    class ServerAddr(object):
        def __init__(self,
            hostname: str,
            remote_port: int,
            ssh_tunnel: bool = False,
            local_port: int | None = None,
            intermediate_host: str | None = None
        ):
            self.host = self.remote_host = hostname
            self.port = self.remote_port = remote_port
            self.ssh_tunnel = ssh_tunnel
            self.local_port = local_port
            self.intermediate_host = intermediate_host
        
        def __repr__(self) -> str:
            if self.ssh_tunnel:
                return f"Addr('{Config.LOCALHOST}':{self.local_port} -> {self.intermediate_host}:{Config.DEFAULT_SSH_PORT} -> {self.remote_host}:{self.remote_port})"

            else:
                return f"Addr('{self.host}':{self.port})"

log = logging.getLogger("pathways-predictions")
