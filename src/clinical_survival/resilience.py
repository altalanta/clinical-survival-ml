"""
Resilience patterns for external service calls.

This module provides:
- Retry with exponential backoff for transient failures
- Circuit breaker pattern to prevent cascading failures
- Graceful degradation for non-critical services
- Configurable timeouts for external calls

Usage:
    from clinical_survival.resilience import (
        retry_with_backoff,
        circuit_breaker,
        with_timeout,
        graceful_degradation,
        CircuitBreaker,
    )
    
    @retry_with_backoff(max_attempts=3)
    def call_external_api():
        ...
        
    @circuit_breaker(failure_threshold=5, recovery_timeout=60)
    def call_mlflow():
        ...
"""

from __future__ import annotations

import functools
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, Set, Tuple, Type, TypeVar, Union
import signal

from clinical_survival.logging_config import get_logger

# Get module logger
logger = get_logger(__name__)

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Retry Pattern
# =============================================================================


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""
    
    def __init__(
        self,
        message: str,
        attempts: int,
        last_exception: Optional[Exception] = None,
    ):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(message)


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[F], F]:
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts (including first try)
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for delay after each attempt
        jitter: Random jitter factor (0-1) to add to delay
        exceptions: Exception types to catch and retry
        on_retry: Optional callback called on each retry with (exception, attempt)
        
    Returns:
        Decorated function that retries on failure
        
    Example:
        @retry_with_backoff(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
        def fetch_data(url: str) -> dict:
            return requests.get(url).json()
    """
    import random
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception: Optional[Exception] = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"All {max_attempts} retry attempts exhausted for {func.__name__}",
                            extra={
                                "function": func.__name__,
                                "attempts": attempt,
                                "error": str(e),
                            },
                        )
                        raise RetryExhaustedError(
                            f"Failed after {max_attempts} attempts: {e}",
                            attempts=attempt,
                            last_exception=e,
                        ) from e
                    
                    # Calculate delay with jitter
                    actual_delay = delay * (1 + random.uniform(-jitter, jitter))
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {actual_delay:.1f}s...",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt,
                            "delay": actual_delay,
                            "error_type": type(e).__name__,
                        },
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt)
                    
                    time.sleep(actual_delay)
                    
                    # Increase delay for next attempt
                    delay = min(delay * backoff_factor, max_delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Circuit Breaker Pattern
# =============================================================================


class CircuitState(Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerState:
    """Internal state for a circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.now)


class CircuitBreaker:
    """
    Circuit breaker implementation for protecting against cascading failures.
    
    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail immediately
    - HALF_OPEN: Testing if the service has recovered
    
    Usage:
        breaker = CircuitBreaker(
            name="mlflow",
            failure_threshold=5,
            recovery_timeout=60,
        )
        
        try:
            with breaker:
                call_mlflow()
        except CircuitOpenError:
            # Handle circuit open state
            use_fallback()
    """
    
    # Class-level registry of circuit breakers
    _instances: Dict[str, "CircuitBreaker"] = {}
    _lock = threading.Lock()
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        recovery_timeout: float = 60.0,
        excluded_exceptions: Optional[Set[Type[Exception]]] = None,
    ):
        """
        Initialize a circuit breaker.
        
        Args:
            name: Unique name for this circuit breaker
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes in half-open to close circuit
            recovery_timeout: Seconds to wait before trying again after opening
            excluded_exceptions: Exceptions that should not count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout)
        self.excluded_exceptions = excluded_exceptions or set()
        self._state = CircuitBreakerState()
        self._lock = threading.Lock()
        
        # Register instance
        with CircuitBreaker._lock:
            CircuitBreaker._instances[name] = self
    
    @classmethod
    def get(cls, name: str) -> Optional["CircuitBreaker"]:
        """Get a circuit breaker by name."""
        return cls._instances.get(name)
    
    @classmethod
    def get_all_states(cls) -> Dict[str, CircuitState]:
        """Get the state of all circuit breakers."""
        return {name: cb._state.state for name, cb in cls._instances.items()}
    
    @property
    def state(self) -> CircuitState:
        """Get the current circuit state."""
        return self._state.state
    
    @property
    def is_closed(self) -> bool:
        """Check if the circuit is closed (normal operation)."""
        return self._state.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if the circuit is open (failing fast)."""
        return self._state.state == CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt a reset."""
        if self._state.last_failure_time is None:
            return True
        return datetime.now() - self._state.last_failure_time >= self.recovery_timeout
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state.state
        self._state.state = new_state
        self._state.last_state_change = datetime.now()
        
        if new_state == CircuitState.CLOSED:
            self._state.failure_count = 0
            self._state.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._state.success_count = 0
        
        logger.info(
            f"Circuit breaker '{self.name}' transitioned: {old_state.value} -> {new_state.value}",
            extra={
                "circuit": self.name,
                "old_state": old_state.value,
                "new_state": new_state.value,
            },
        )
    
    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state.state == CircuitState.HALF_OPEN:
                self._state.success_count += 1
                if self._state.success_count >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
    
    def record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        # Check if this exception should be excluded
        if type(exception) in self.excluded_exceptions:
            return
        
        with self._lock:
            self._state.failure_count += 1
            self._state.last_failure_time = datetime.now()
            
            if self._state.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                self._transition_to(CircuitState.OPEN)
            elif self._state.state == CircuitState.CLOSED:
                if self._state.failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def __enter__(self) -> "CircuitBreaker":
        """Enter the circuit breaker context."""
        with self._lock:
            if self._state.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' is open",
                        circuit_name=self.name,
                        recovery_time=self.recovery_timeout.total_seconds(),
                    )
        return self
    
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Exit the circuit breaker context."""
        if exc_val is None:
            self.record_success()
        elif isinstance(exc_val, Exception):
            self.record_failure(exc_val)
        return False  # Don't suppress exceptions
    
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Call a function through the circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If the circuit is open
        """
        with self:
            return func(*args, **kwargs)
    
    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open."""
    
    def __init__(
        self,
        message: str,
        circuit_name: str,
        recovery_time: float,
    ):
        self.circuit_name = circuit_name
        self.recovery_time = recovery_time
        super().__init__(message)


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    recovery_timeout: float = 60.0,
    fallback: Optional[Callable[..., Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator to wrap a function with a circuit breaker.
    
    Args:
        name: Circuit breaker name (defaults to function name)
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes in half-open to close
        recovery_timeout: Seconds to wait before trying again
        fallback: Optional fallback function to call when circuit is open
        
    Returns:
        Decorated function
        
    Example:
        @circuit_breaker(name="mlflow", failure_threshold=3, fallback=lambda: None)
        def log_to_mlflow(metrics):
            mlflow.log_metrics(metrics)
    """
    def decorator(func: F) -> F:
        breaker_name = name or func.__name__
        breaker = CircuitBreaker(
            name=breaker_name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            recovery_timeout=recovery_timeout,
        )
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return breaker.call(func, *args, **kwargs)
            except CircuitOpenError:
                if fallback is not None:
                    logger.warning(
                        f"Circuit '{breaker_name}' is open, using fallback",
                        extra={"circuit": breaker_name},
                    )
                    return fallback(*args, **kwargs)
                raise
        
        # Attach breaker to function for inspection
        wrapper.circuit_breaker = breaker  # type: ignore
        return wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Timeout Pattern
# =============================================================================


class TimeoutError(Exception):
    """Raised when an operation times out."""
    
    def __init__(self, message: str, timeout: float):
        self.timeout = timeout
        super().__init__(message)


@contextmanager
def timeout_context(seconds: float, operation: str = "operation"):
    """
    Context manager to enforce a timeout on an operation.
    
    Note: This uses SIGALRM and only works on Unix systems.
    On Windows, it will log a warning and not enforce the timeout.
    
    Args:
        seconds: Timeout in seconds
        operation: Description of the operation (for error messages)
        
    Yields:
        None
        
    Raises:
        TimeoutError: If the operation exceeds the timeout
        
    Example:
        with timeout_context(30, "MLflow logging"):
            mlflow.log_metrics(metrics)
    """
    import platform
    
    if platform.system() == "Windows":
        logger.warning(
            f"Timeout not supported on Windows for '{operation}'",
            extra={"operation": operation, "timeout": seconds},
        )
        yield
        return
    
    def timeout_handler(signum: int, frame: Any) -> None:
        raise TimeoutError(
            f"Operation '{operation}' timed out after {seconds}s",
            timeout=seconds,
        )
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    
    try:
        yield
    finally:
        # Cancel the alarm and restore the old handler
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def with_timeout(
    seconds: float,
    fallback: Optional[Callable[..., Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator to enforce a timeout on a function.
    
    Args:
        seconds: Timeout in seconds
        fallback: Optional fallback function to call on timeout
        
    Returns:
        Decorated function
        
    Example:
        @with_timeout(30, fallback=lambda: {})
        def fetch_external_data():
            return requests.get(url).json()
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                with timeout_context(seconds, func.__name__):
                    return func(*args, **kwargs)
            except TimeoutError:
                if fallback is not None:
                    logger.warning(
                        f"Function '{func.__name__}' timed out, using fallback",
                        extra={"function": func.__name__, "timeout": seconds},
                    )
                    return fallback(*args, **kwargs)
                raise
        
        return wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Graceful Degradation
# =============================================================================


class GracefulDegradation(Generic[T]):
    """
    Wrapper for graceful degradation of non-critical services.
    
    Allows operations to fail silently while logging the failure,
    returning a default value instead of raising an exception.
    
    Usage:
        tracker = GracefulDegradation(
            MLflowTracker(config),
            default_factory=lambda: NullTracker(),
            name="mlflow",
        )
        
        # This won't raise even if MLflow is down
        tracker.log_metrics({"accuracy": 0.95})
    """
    
    def __init__(
        self,
        service: T,
        default_factory: Optional[Callable[[], T]] = None,
        name: str = "service",
        log_failures: bool = True,
    ):
        """
        Initialize graceful degradation wrapper.
        
        Args:
            service: The service to wrap
            default_factory: Factory to create default/fallback service
            name: Name for logging
            log_failures: Whether to log failures
        """
        self._service = service
        self._default_factory = default_factory
        self._name = name
        self._log_failures = log_failures
        self._is_degraded = False
        self._failure_count = 0
    
    @property
    def is_degraded(self) -> bool:
        """Check if the service is currently degraded."""
        return self._is_degraded
    
    @property
    def failure_count(self) -> int:
        """Get the number of failures."""
        return self._failure_count
    
    def _get_fallback(self) -> Optional[T]:
        """Get the fallback service."""
        if self._default_factory:
            return self._default_factory()
        return None
    
    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the wrapped service with error handling."""
        attr = getattr(self._service, name)
        
        if not callable(attr):
            return attr
        
        @functools.wraps(attr)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = attr(*args, **kwargs)
                # Reset degraded state on success
                if self._is_degraded:
                    self._is_degraded = False
                    logger.info(
                        f"Service '{self._name}' recovered",
                        extra={"service": self._name},
                    )
                return result
            except Exception as e:
                self._failure_count += 1
                self._is_degraded = True
                
                if self._log_failures:
                    logger.warning(
                        f"Service '{self._name}' degraded: {e}",
                        extra={
                            "service": self._name,
                            "method": name,
                            "failure_count": self._failure_count,
                            "error_type": type(e).__name__,
                        },
                    )
                
                # Try fallback
                fallback = self._get_fallback()
                if fallback is not None:
                    fallback_attr = getattr(fallback, name, None)
                    if callable(fallback_attr):
                        return fallback_attr(*args, **kwargs)
                
                return None
        
        return wrapper


def graceful_degradation(
    default: Any = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_level: str = "warning",
) -> Callable[[F], F]:
    """
    Decorator for graceful degradation of non-critical operations.
    
    Args:
        default: Default value to return on failure
        exceptions: Exception types to catch
        log_level: Log level for failures
        
    Returns:
        Decorated function
        
    Example:
        @graceful_degradation(default=None)
        def log_to_external_service(data):
            external_api.log(data)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                log_method = getattr(logger, log_level)
                log_method(
                    f"Graceful degradation for '{func.__name__}': {e}",
                    extra={
                        "function": func.__name__,
                        "error_type": type(e).__name__,
                    },
                )
                return default
        
        return wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Combined Resilience Decorator
# =============================================================================


def resilient(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    circuit_name: Optional[str] = None,
    circuit_threshold: int = 5,
    timeout_seconds: Optional[float] = None,
    fallback: Optional[Callable[..., Any]] = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Combined resilience decorator with retry, circuit breaker, and timeout.
    
    This is a convenience decorator that combines multiple resilience patterns.
    
    Args:
        max_retries: Maximum retry attempts
        retry_delay: Initial retry delay
        circuit_name: Circuit breaker name (enables circuit breaker if set)
        circuit_threshold: Failures before opening circuit
        timeout_seconds: Timeout in seconds (enables timeout if set)
        fallback: Fallback function on failure
        exceptions: Exceptions to handle
        
    Returns:
        Decorated function with resilience patterns
        
    Example:
        @resilient(
            max_retries=3,
            circuit_name="external_api",
            timeout_seconds=30,
            fallback=lambda: {"status": "unavailable"},
        )
        def call_external_api():
            return requests.get(url).json()
    """
    def decorator(func: F) -> F:
        # Start with the base function
        wrapped = func
        
        # Apply timeout if specified
        if timeout_seconds is not None:
            wrapped = with_timeout(timeout_seconds)(wrapped)
        
        # Apply retry
        wrapped = retry_with_backoff(
            max_attempts=max_retries,
            initial_delay=retry_delay,
            exceptions=exceptions,
        )(wrapped)
        
        # Apply circuit breaker if specified
        if circuit_name is not None:
            wrapped = circuit_breaker(
                name=circuit_name,
                failure_threshold=circuit_threshold,
                fallback=fallback,
            )(wrapped)
        elif fallback is not None:
            # Apply graceful degradation if no circuit breaker but fallback specified
            @functools.wraps(wrapped)
            def with_fallback(*args: Any, **kwargs: Any) -> Any:
                try:
                    return wrapped(*args, **kwargs)
                except Exception:
                    return fallback(*args, **kwargs)
            wrapped = with_fallback
        
        return wrapped  # type: ignore
    
    return decorator











