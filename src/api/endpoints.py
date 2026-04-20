"""
API Endpoints for PrometheusDB.

Reusable endpoint components for building custom API routes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class RequestHandler(Protocol):
    """Protocol for request handlers."""
    async def __call__(self, request: Any) -> Any: ...


@dataclass
class Endpoint:
    """An API endpoint definition."""
    path: str
    method: str
    handler: RequestHandler
    description: str = ""
    tags: list[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class Router:
    """
    Simple router for API endpoints.
    
    Example:
        >>> router = Router()
        >>> @router.get("/metrics")
        ... async def get_metrics():
        ...     return {"metrics": []}
        >>> app = router.to_app()
    """
    
    def __init__(self):
        """Initialize router."""
        self._endpoints: list[Endpoint] = []
        self._middlewares: list[Callable] = []
    
    def get(self, path: str, **kwargs) -> Callable:
        """Register GET endpoint."""
        def decorator(handler: RequestHandler) -> RequestHandler:
            self._endpoints.append(Endpoint(
                path=path,
                method="GET",
                handler=handler,
                **kwargs
            ))
            return handler
        return decorator
    
    def post(self, path: str, **kwargs) -> Callable:
        """Register POST endpoint."""
        def decorator(handler: RequestHandler) -> RequestHandler:
            self._endpoints.append(Endpoint(
                path=path,
                method="POST",
                handler=handler,
                **kwargs
            ))
            return handler
        return decorator
    
    def put(self, path: str, **kwargs) -> Callable:
        """Register PUT endpoint."""
        def decorator(handler: RequestHandler) -> RequestHandler:
            self._endpoints.append(Endpoint(
                path=path,
                method="PUT",
                handler=handler,
                **kwargs
            ))
            return handler
        return decorator
    
    def delete(self, path: str, **kwargs) -> Callable:
        """Register DELETE endpoint."""
        def decorator(handler: RequestHandler) -> RequestHandler:
            self._endpoints.append(Endpoint(
                path=path,
                method="DELETE",
                handler=handler,
                **kwargs
            ))
            return handler
        return decorator
    
    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware function."""
        self._middlewares.append(middleware)
    
    def to_app(self, **kwargs) -> Any:
        """Convert router to FastAPI app."""
        try:
            from fastapi import FastAPI
        except ImportError:
            raise ImportError("FastAPI required")
        
        app = FastAPI(**kwargs)
        
        for endpoint in self._endpoints:
            methods = [endpoint.method]
            
            if endpoint.method == "GET":
                app.get(endpoint.path, tags=endpoint.tags)(endpoint.handler)
            elif endpoint.method == "POST":
                app.post(endpoint.path, tags=endpoint.tags)(endpoint.handler)
            elif endpoint.method == "PUT":
                app.put(endpoint.path, tags=endpoint.tags)(endpoint.handler)
            elif endpoint.method == "DELETE":
                app.delete(endpoint.path, tags=endpoint.tags)(endpoint.handler)
        
        for middleware in self._middlewares:
            app.middleware("http")(middleware)
        
        return app


class APIRouter:
    """
    Grouped API router with common configuration.
    
    Example:
        >>> metrics_router = APIRouter(prefix="/metrics", tags=["Metrics"])
        >>> @metrics_router.get("/")
        ... async def list_metrics():
        ...     return []
    """
    
    def __init__(
        self,
        prefix: str = "",
        tags: list[str] = None,
        dependencies: list = None,
    ):
        """Initialize API router."""
        self.prefix = prefix
        self.tags = tags or []
        self.dependencies = dependencies or []
        self._routes = []
    
    def get(self, path: str, **kwargs) -> Callable:
        """Register GET endpoint."""
        def decorator(handler: RequestHandler) -> RequestHandler:
            self._routes.append({
                "path": self.prefix + path,
                "method": "GET",
                "handler": handler,
                "kwargs": kwargs
            })
            return handler
        return decorator
    
    def post(self, path: str, **kwargs) -> Callable:
        """Register POST endpoint."""
        def decorator(handler: RequestHandler) -> RequestHandler:
            self._routes.append({
                "path": self.prefix + path,
                "method": "POST",
                "handler": handler,
                "kwargs": kwargs
            })
            return handler
        return decorator
    
    def put(self, path: str, **kwargs) -> Callable:
        """Register PUT endpoint."""
        def decorator(handler: RequestHandler) -> RequestHandler:
            self._routes.append({
                "path": self.prefix + path,
                "method": "PUT",
                "handler": handler,
                "kwargs": kwargs
            })
            return handler
        return decorator
    
    def delete(self, path: str, **kwargs) -> Callable:
        """Register DELETE endpoint."""
        def decorator(handler: RequestHandler) -> RequestHandler:
            self._routes.append({
                "path": self.prefix + path,
                "method": "DELETE",
                "handler": handler,
                "kwargs": kwargs
            })
            return handler
        return decorator
    
    def include_in(self, app: Any) -> None:
        """Include routes in FastAPI app."""
        try:
            from fastapi import APIRouter as FastAPIRouter
        except ImportError:
            raise ImportError("FastAPI required")
        
        router = FastAPIRouter(
            prefix=self.prefix,
            tags=self.tags,
            dependencies=self.dependencies,
        )
        
        for route in self._routes:
            method = route["method"].lower()
            getattr(router, method)(
                route["path"].replace(self.prefix, ""),
                **route.get("kwargs", {})
            )(route["handler"])
        
        app.include_router(router)
