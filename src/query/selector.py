"""
Label and Time Series Selector Implementation.

This module provides the selector matching logic for PromQL queries,
including support for various label matching operators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class MatchOperator(Enum):
    """Label matching operators."""
    EQUAL = "="           # Exact match
    NOT_EQUAL = "!="      # Not equal
    REGEX_MATCH = "=~"     # Regex match
    REGEX_NOT_MATCH = "!~" # Regex not match


@dataclass
class LabelSelector:
    """
    Label selector for filtering time series.
    
    Supports exact matching, negation, and regex patterns.
    
    Example:
        >>> selector = LabelSelector("method", MatchOperator.EQUAL, "GET")
        >>> selector.matches({"method": "GET", "status": "200"})
        True
        >>> selector.matches({"method": "POST", "status": "200"})
        False
    """
    name: str
    operator: MatchOperator
    value: str
    _pattern: re.Pattern | None = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        if self.operator in (MatchOperator.REGEX_MATCH, MatchOperator.REGEX_NOT_MATCH):
            try:
                self._pattern = re.compile(self.value)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{self.value}': {e}")
    
    def matches(self, labels: dict[str, str]) -> bool:
        """
        Check if labels match this selector.
        
        Args:
            labels: Dictionary of label name -> value
        
        Returns:
            True if labels match the selector
        """
        label_value = labels.get(self.name)
        
        if label_value is None:
            return self.operator == MatchOperator.NOT_EQUAL
        
        if self.operator == MatchOperator.EQUAL:
            return label_value == self.value
        elif self.operator == MatchOperator.NOT_EQUAL:
            return label_value != self.value
        elif self.operator == MatchOperator.REGEX_MATCH:
            return self._pattern is not None and bool(self._pattern.match(label_value))
        elif self.operator == MatchOperator.REGEX_NOT_MATCH:
            return self._pattern is None or not self._pattern.match(label_value)
        
        return False
    
    def __str__(self) -> str:
        return f"{self.name}{self.operator.value}\"{self.value}\""
    
    def __repr__(self) -> str:
        return f"LabelSelector({self.name}, {self.operator.value}, {self.value})"


@dataclass
class TimeSeriesSelector:
    """
    Selector for time series including metric name and labels.
    
    This combines metric name matching with label selectors
    to identify specific time series.
    
    Example:
        >>> selector = TimeSeriesSelector(
        ...     metric_name="http_requests_total",
        ...     labels=[
        ...         LabelSelector("method", MatchOperator.EQUAL, "GET"),
        ...         LabelSelector("status", MatchOperator.REGEX_MATCH, "2.."),
        ...     ]
        ... )
        >>> selector.matches(metric_name="http_requests_total", labels={"method": "GET", "status": "200"})
        True
    """
    metric_name: str
    labels: list[LabelSelector] = field(default_factory=list)
    offset_ms: int = 0
    
    @classmethod
    def parse(cls, selector_str: str) -> TimeSeriesSelector:
        """
        Parse a selector string.
        
        Args:
            selector_str: Selector string like 'metric_name{label="value"}'
        
        Returns:
            TimeSeriesSelector instance
        
        Example:
            >>> selector = TimeSeriesSelector.parse('http_requests_total{method="GET"}')
        """
        # Parse metric name
        metric_pattern = re.match(r'^([a-zA-Z_][a-zA-Z0-9_:]*)', selector_str)
        if not metric_pattern:
            raise ValueError(f"Invalid selector string: {selector_str}")
        
        metric_name = metric_pattern.group(1)
        remaining = selector_str[metric_pattern.end():]
        
        # Parse labels
        labels = []
        
        if remaining.startswith('{'):
            remaining = remaining[1:]
            
            while remaining and remaining[0] != '}':
                # Parse label name
                label_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*', remaining)
                if not label_match:
                    raise ValueError(f"Invalid label in selector: {remaining}")
                
                label_name = label_match.group(1)
                remaining = remaining[label_match.end():]
                
                # Parse operator
                if remaining.startswith('=~'):
                    op = MatchOperator.REGEX_MATCH
                    remaining = remaining[2:]
                elif remaining.startswith('!~'):
                    op = MatchOperator.REGEX_NOT_MATCH
                    remaining = remaining[2:]
                elif remaining.startswith('!='):
                    op = MatchOperator.NOT_EQUAL
                    remaining = remaining[2:]
                elif remaining.startswith('='):
                    op = MatchOperator.EQUAL
                    remaining = remaining[1:]
                else:
                    raise ValueError(f"Expected operator, got: {remaining}")
                
                # Parse value
                if remaining[0] == '"':
                    # Double-quoted string
                    end = remaining.find('"', 1)
                    if end == -1:
                        raise ValueError("Unterminated string")
                    value = remaining[1:end]
                    remaining = remaining[end + 1:]
                elif remaining[0] == "'":
                    # Single-quoted string
                    end = remaining.find("'", 1)
                    if end == -1:
                        raise ValueError("Unterminated string")
                    value = remaining[1:end]
                    remaining = remaining[end + 1:]
                else:
                    raise ValueError(f"Expected string, got: {remaining}")
                
                labels.append(LabelSelector(name=label_name, operator=op, value=value))
                
                # Skip comma
                if remaining.startswith(','):
                    remaining = remaining[1:]
            
            # Expect closing brace
            if remaining[0] == '}':
                remaining = remaining[1:]
        
        return cls(metric_name=metric_name, labels=labels)
    
    def matches(
        self,
        metric_name: str,
        labels: dict[str, str],
    ) -> bool:
        """
        Check if a metric matches this selector.
        
        Args:
            metric_name: Metric name to match
            labels: Label dictionary
        
        Returns:
            True if the metric matches all selectors
        """
        # Match metric name (supports regex)
        if self.metric_name and not self._match_metric_name(metric_name):
            return False
        
        # Match all labels
        return all(selector.matches(labels) for selector in self.labels)
    
    def _match_metric_name(self, name: str) -> bool:
        """Match metric name (supports regex)."""
        if ':' in self.metric_name:
            # Label matcher format: metric_name:label_name=value
            parts = self.metric_name.split(':')
            base_name = parts[0]
            if not name.startswith(base_name):
                return False
            
            if len(parts) > 1:
                # Has label matcher
                matcher_parts = parts[1].split('=')
                if len(matcher_parts) == 2:
                    label_name, label_value = matcher_parts
                    return labels.get(label_name) == label_value
            
            return True
        
        return name == self.metric_name or name.startswith(self.metric_name + '{')
    
    def __str__(self) -> str:
        labels_str = ",".join(str(l) for l in self.labels)
        return f"{self.metric_name}{{{labels_str}}}"
    
    def __repr__(self) -> str:
        return f"TimeSeriesSelector({self.metric_name}, labels={self.labels})"


def label_matchers_match(
    selectors: list[LabelSelector],
    labels: dict[str, str],
) -> bool:
    """
    Check if labels match all selectors.
    
    This is a utility function for quick matching.
    
    Args:
        selectors: List of LabelSelector objects
        labels: Label dictionary to match against
    
    Returns:
        True if all selectors match
    """
    return all(selector.matches(labels) for selector in selectors)


def parse_label_matchers(matchers: list[tuple[str, str, str]]) -> list[LabelSelector]:
    """
    Parse label matchers from tuples.
    
    Args:
        matchers: List of (name, operator, value) tuples
    
    Returns:
        List of LabelSelector objects
    """
    operators = {
        '=': MatchOperator.EQUAL,
        '!=': MatchOperator.NOT_EQUAL,
        '=~': MatchOperator.REGEX_MATCH,
        '!~': MatchOperator.REGEX_NOT_MATCH,
    }
    
    return [
        LabelSelector(
            name=name,
            operator=operators.get(op, MatchOperator.EQUAL),
            value=value
        )
        for name, op, value in matchers
    ]
