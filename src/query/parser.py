"""
PromQL Parser - Lexer and Parser for PromQL Queries.

This module implements a full PromQL parser that tokenizes and parses
PromQL expressions into an AST for execution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class TokenType(Enum):
    """PromQL token types."""
    METRIC_NAME = auto()
    LABEL_NAME = auto()
    LABEL_VALUE = auto()
    NUMBER = auto()
    DURATION = auto()
    OPERATOR = auto()
    COMPARISON = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    COLON = auto()
    EQUAL = auto()
    BANG = auto()
    PREFIX = auto()
    AGGREGATION = auto()
    FUNCTION = auto()
    KEYWORD = auto()
    EOF = auto()


@dataclass
class Token:
    """A parsed token."""
    type: TokenType
    value: str
    position: int = 0
    length: int = 0


class Lexer:
    """
    Lexer for PromQL tokenization.
    
    Converts a PromQL query string into a stream of tokens.
    """
    
    # Token patterns
    PATTERNS = [
        (r'[a-zA-Z_][a-zA-Z0-9_]*:', TokenType.PREFIX),  # Label matcher prefix
        (r'(sum|avg|min|max|count|stddev|stdvar|stdvar|bottomk|topk|'
         r'count_values|quantile)\b', TokenType.AGGREGATION),
        (r'(rate|increase|irate|avg_over_time|sum_over_time|min_over_time|'
         r'max_over_time|count_over_time|stddev_over_time|absent|abs|ceil|'
         r'floor|round|sqrt|exp|ln|log2|log10|scalar|vector|clamp_max|'
         r'clamp_min|histogram_quantile|predict_linear|histogram_fraction)\b', 
         TokenType.FUNCTION),
        (r'(by|on|without|ignoring|group_left|group_right)\b', TokenType.KEYWORD),
        (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.METRIC_NAME),
        (r'[a-zA-Z_][a-zA-Z0-9_]*\s*(=|!=|=~|!~)', TokenType.LABEL_NAME),
        (r'"[^"]*"', TokenType.LABEL_VALUE),
        (r'[\d]+\.?[\d]*', TokenType.NUMBER),
        (r'[\d]+[smhdwy]', TokenType.DURATION),
        (r'[\+\-\*/\%\^]', TokenType.OPERATOR),
        (r'[<>]=?|==?|!=', TokenType.COMPARISON),
        (r'[\(\)]', TokenType.LPAREN if r'(' else TokenType.RPAREN),
        (r'\[\]', TokenType.LBRACKET),
        (r'\{', TokenType.LBRACE),
        (r'\}', TokenType.RBRACE),
        (r',', TokenType.COMMA),
        (r':', TokenType.COLON),
    ]
    
    def __init__(self, query: str):
        """Initialize lexer with input query."""
        self.query = query
        self.pos = 0
        self.tokens: list[Token] = []
    
    def tokenize(self) -> list[Token]:
        """Tokenize the entire query."""
        while self.pos < len(self.query):
            self._skip_whitespace()
            if self.pos >= len(self.query):
                break
            
            matched = False
            for pattern, token_type in self.PATTERNS:
                regex = re.compile(pattern)
                match = regex.match(self.query, self.pos)
                if match:
                    value = match.group()
                    self.tokens.append(Token(
                        type=token_type,
                        value=value,
                        position=self.pos,
                        length=len(value)
                    ))
                    self.pos = match.end()
                    matched = True
                    break
            
            if not matched:
                raise ValueError(f"Unexpected character at position {self.pos}: '{self.query[self.pos]}'")
        
        self.tokens.append(Token(type=TokenType.EOF, value="", position=self.pos))
        return self.tokens
    
    def _skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.pos < len(self.query) and self.query[self.pos].isspace():
            self.pos += 1


class ASTNode:
    """Base class for AST nodes."""
    pass


@dataclass
class MetricNode(ASTNode):
    """Metric name node."""
    name: str
    labels: dict[str, tuple[str, str]] = field(default_factory=dict)  # name -> (op, value)


@dataclass
class LabelMatcherNode(ASTNode):
    """Label matcher node."""
    name: str
    op: str
    value: str


@dataclass
class VectorSelectorNode(ASTNode):
    """Vector selector node (metric name + labels)."""
    metric: MetricNode
    start: int = 0
    end: int = 0
    offset: int = 0


@dataclass
class RangeSelectorNode(ASTNode):
    """Range vector selector."""
    vector_selector: VectorSelectorNode
    duration_ms: int


@dataclass
class FunctionCallNode(ASTNode):
    """Function call node."""
    name: str
    args: list[ASTNode]


@dataclass
class AggregationNode(ASTNode):
    """Aggregation operator node."""
    operator: str
    expr: ASTNode
    group_by: list[str] = field(default_factory=list)
    group_without: bool = False
    labels: list[tuple[str, str]] = field(default_factory=list)  # (name, value) for group_left/right


@dataclass
class BinaryOpNode(ASTNode):
    """Binary operation node."""
    op: str
    left: ASTNode
    right: ASTNode
    matching: str | None = None  # 'ignoring', 'on'
    card: str | None = None  # 'group_left', 'group_right'


@dataclass
class NumberLiteralNode(ASTNode):
    """Number literal node."""
    value: float


@dataclass
class StringLiteralNode(ASTNode):
    """String literal node."""
    value: str


class PromQLParser:
    """
    Parser for PromQL expressions.
    
    Parses PromQL queries into an Abstract Syntax Tree (AST)
    that can be executed by the query engine.
    """
    
    def __init__(self, tokens: list[Token]):
        """Initialize parser with tokens."""
        self.tokens = tokens
        self.pos = 0
    
    @property
    def current(self) -> Token:
        """Get current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]
    
    def _advance(self) -> Token:
        """Advance to next token and return current."""
        token = self.current
        self.pos += 1
        return token
    
    def _expect(self, token_type: TokenType) -> Token:
        """Expect a specific token type."""
        if self.current.type != token_type:
            raise ValueError(
                f"Expected {token_type}, got {self.current.type} "
                f"at position {self.current.position}"
            )
        return self._advance()
    
    def _match(self, token_type: TokenType) -> bool:
        """Check if current token matches and advance if so."""
        if self.current.type == token_type:
            self._advance()
            return True
        return False
    
    def parse(self) -> ASTNode:
        """Parse the entire expression."""
        return self._parse_expression()
    
    def _parse_expression(self) -> ASTNode:
        """Parse an expression (may contain binary ops)."""
        return self._parse_additive()
    
    def _parse_additive(self) -> ASTNode:
        """Parse additive expressions (+, -)."""
        left = self._parse_multiplicative()
        
        while self.current.type == TokenType.OPERATOR and self.current.value in '+-':
            op = self._advance().value
            right = self._parse_multiplicative()
            left = BinaryOpNode(op=op, left=left, right=right)
        
        return left
    
    def _parse_multiplicative(self) -> ASTNode:
        """Parse multiplicative expressions (*, /, %, ^)."""
        left = self._parse_unary()
        
        while self.current.type == TokenType.OPERATOR and self.current.value in '*/%^':
            op = self._advance().value
            right = self._parse_unary()
            left = BinaryOpNode(op=op, left=left, right=right)
        
        return left
    
    def _parse_unary(self) -> ASTNode:
        """Parse unary expressions (-, +)."""
        if self.current.type == TokenType.OPERATOR and self.current.value in '+-':
            op = self._advance().value
            expr = self._parse_unary()
            return BinaryOpNode(op=op, left=NumberLiteralNode(0), right=expr)
        
        return self._parse_primary()
    
    def _parse_primary(self) -> ASTNode:
        """Parse primary expressions."""
        # Number literal
        if self.current.type == TokenType.NUMBER:
            value = float(self._advance().value)
            return NumberLiteralNode(value)
        
        # String literal
        if self.current.type == TokenType.LABEL_VALUE:
            value = self._advance().value.strip('"')
            return StringLiteralNode(value)
        
        # Function call
        if self.current.type == TokenType.FUNCTION:
            return self._parse_function_call()
        
        # Aggregation
        if self.current.type == TokenType.AGGREGATION:
            return self._parse_aggregation()
        
        # Vector selector or range selector
        return self._parse_vector_selector()
    
    def _parse_function_call(self) -> FunctionCallNode:
        """Parse function call."""
        name = self._advance().value
        self._expect(TokenType.LPAREN)
        
        args = []
        while self.current.type != TokenType.RPAREN:
            args.append(self._parse_expression())
            if not self._match(TokenType.COMMA):
                break
        
        self._expect(TokenType.RPAREN)
        return FunctionCallNode(name=name, args=args)
    
    def _parse_aggregation(self) -> AggregationNode:
        """Parse aggregation expression."""
        operator = self._advance().value
        
        group_by = []
        group_without = False
        
        # Check for 'by' or 'without'
        if self.current.type == TokenType.KEYWORD:
            keyword = self.current.value
            if keyword == 'by':
                self._advance()
                self._expect(TokenType.LPAREN)
                while self.current.type != TokenType.RPAREN:
                    if self.current.type == TokenType.METRIC_NAME:
                        group_by.append(self._advance().value)
                    if not self._match(TokenType.COMMA):
                        break
                self._expect(TokenType.RPAREN)
            elif keyword == 'without':
                group_without = True
                self._advance()
                self._expect(TokenType.LPAREN)
                while self.current.type != TokenType.RPAREN:
                    if self.current.type == TokenType.METRIC_NAME:
                        group_by.append(self._advance().value)
                    if not self._match(TokenType.COMMA):
                        break
                self._expect(TokenType.RPAREN)
        
        expr = self._parse_primary()
        
        return AggregationNode(
            operator=operator,
            expr=expr,
            group_by=group_by,
            group_without=group_without
        )
    
    def _parse_vector_selector(self) -> ASTNode:
        """Parse vector selector."""
        # Metric name
        metric_name = ""
        if self.current.type == TokenType.METRIC_NAME:
            metric_name = self._advance().value
        
        # Labels
        labels: dict[str, tuple[str, str]] = {}
        
        if self.current.type == TokenType.LBRACE or (
            self.current.type == TokenType.LABEL_NAME
        ):
            if self.current.type == TokenType.LBRACE:
                self._advance()
            
            while self.current.type == TokenType.LABEL_NAME or (
                self.current.type == TokenType.METRIC_NAME and ':' in self.current.value
            ):
                # Parse label matcher
                token = self._advance().value
                if ':' in token:
                    # Format: label_name: or label_name=value
                    parts = token.rsplit(':', 1)
                    name = parts[0]
                    if len(parts) > 1 and parts[1]:
                        op, value = '=', parts[1].strip('"')
                        labels[name] = (op, value)
                else:
                    name = token
                    op = self._advance().value  # =, !=, =~, !~
                    value = self._advance().value.strip('"')
                    labels[name] = (op, value)
                
                if not self._match(TokenType.COMMA):
                    break
            
            if self.current.type == TokenType.RBRACE:
                self._advance()
        
        metric = MetricNode(name=metric_name, labels=labels)
        
        # Check for range selector [5m], [1h]
        if self.current.type == TokenType.LBRACKET:
            self._advance()
            duration_str = self._expect(TokenType.DURATION).value
            self._expect(TokenType.RBRACKET)
            
            # Parse duration
            multipliers = {'s': 1000, 'm': 60000, 'h': 3600000, 'd': 86400000, 'w': 604800000}
            duration_ms = int(duration_str[:-1]) * multipliers.get(duration_str[-1], 1)
            
            vector_selector = VectorSelectorNode(metric=metric)
            return RangeSelectorNode(vector_selector=vector_selector, duration_ms=duration_ms)
        
        return VectorSelectorNode(metric=metric)
    
    def _parse_comparison(self) -> ASTNode:
        """Parse comparison operators."""
        left = self._parse_expression()
        
        if self.current.type == TokenType.COMPARISON:
            op = self._advance().value
            right = self._parse_expression()
            return BinaryOpNode(op=op, left=left, right=right)
        
        return left


def parse_query(query: str) -> ASTNode:
    """
    Parse a PromQL query string into an AST.
    
    Args:
        query: PromQL query string
    
    Returns:
        Root AST node
    
    Example:
        >>> ast = parse_query('rate(http_requests_total[5m])')
        >>> ast = parse_query('sum by (method) (rate(http_requests_total[5m]))')
    """
    lexer = Lexer(query)
    tokens = lexer.tokenize()
    parser = PromQLParser(tokens)
    return parser.parse()


def ast_to_string(node: ASTNode, indent: int = 0) -> str:
    """Convert AST back to string representation (for debugging)."""
    prefix = "  " * indent
    
    if isinstance(node, MetricNode):
        return f"{prefix}Metric({node.name}, labels={node.labels})"
    elif isinstance(node, VectorSelectorNode):
        return f"{prefix}VectorSelector({ast_to_string(node.metric, 0)})"
    elif isinstance(node, RangeSelectorNode):
        return (f"{prefix}RangeSelector("
                f"selector={ast_to_string(node.vector_selector, 0)}, "
                f"duration={node.duration_ms}ms)")
    elif isinstance(node, FunctionCallNode):
        args = ", ".join(ast_to_string(a, 0) for a in node.args)
        return f"{prefix}FunctionCall({node.name}, [{args}])"
    elif isinstance(node, AggregationNode):
        return (f"{prefix}Aggregation({node.operator}, "
                f"expr={ast_to_string(node.expr, 0)}, "
                f"group_by={node.group_by})")
    elif isinstance(node, BinaryOpNode):
        return (f"{prefix}BinaryOp({node.op}, "
                f"left={ast_to_string(node.left, 0)}, "
                f"right={ast_to_string(node.right, 0)})")
    elif isinstance(node, NumberLiteralNode):
        return f"{prefix}Number({node.value})"
    elif isinstance(node, StringLiteralNode):
        return f"{prefix}String({node.value})"
    else:
        return f"{prefix}Unknown({type(node).__name__})"
