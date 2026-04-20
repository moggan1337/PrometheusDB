"""
HTTP API Server for PrometheusDB.

Provides REST API for:
- Writing metrics
- Querying data
- Managing retention policies
- Cluster operations
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse, Response
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 9090
    debug: bool = False
    cors_enabled: bool = True
    cors_origins: list[str] = None
    max_request_size: int = 10 * 1024 * 1024  # 10MB


class APIServer:
    """
    HTTP API server for PrometheusDB.
    
    Provides endpoints for:
    - POST /write - Write metrics
    - GET /query - Query metrics
    - GET /series - List series
    - GET /label_values - Get label values
    - POST /admin/retention - Manage retention
    - GET /stats - Server statistics
    """
    
    def __init__(
        self,
        database: Any,
        config: APIConfig | None = None,
    ):
        """
        Initialize API server.
        
        Args:
            database: PrometheusDB instance
            config: API configuration
        """
        self.db = database
        self.config = config or APIConfig()
        
        self.app: Any = None
        self._setup_app()
    
    def _setup_app(self) -> None:
        """Set up FastAPI application."""
        if not FASTAPI_AVAILABLE:
            print("Warning: FastAPI not installed. API server unavailable.")
            return
        
        self.app = FastAPI(
            title="PrometheusDB API",
            description="High-performance Time-Series Database with Vector Operations",
            version="0.1.0",
        )
        
        # CORS middleware
        if self.config.cors_enabled:
            origins = self.config.cors_origins or ["*"]
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Add routes
        self._add_routes()
    
    def _add_routes(self) -> None:
        """Add API routes."""
        
        @self.app.middleware("http")
        async def add_process_time(request: Request, call_next):
            """Add processing time header."""
            start = time.time()
            response = await call_next(request)
            response.headers["X-Process-Time"] = str(time.time() - start)
            return response
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "version": "0.1.0"}
        
        @self.app.get("/stats")
        async def stats():
            """Get database statistics."""
            return self.db.get_stats()
        
        @self.app.post("/write")
        async def write_metric(data: dict[str, Any]):
            """Write a metric data point."""
            try:
                self.db.write(
                    metric_name=data["metric_name"],
                    labels=data.get("labels", {}),
                    value=data["value"],
                    timestamp=data.get("timestamp"),
                )
                return {"success": True}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/write/batch")
        async def write_batch(data: dict[str, Any]):
            """Write multiple metric data points."""
            try:
                count = self.db.write_batch(data.get("data", []))
                return {"success": True, "count": count}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/query")
        async def query(
            query: str,
            time: int | None = None,
        ):
            """Execute a PromQL query."""
            try:
                results = self.db.query(query, time)
                return {
                    "results": [r.to_dict() for r in results]
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/query_range")
        async def query_range(
            query: str,
            start: int,
            end: int,
            step: int = 15000,
        ):
            """Execute a range query."""
            try:
                results = self.db.query(query, end)
                return {
                    "query": query,
                    "start": start,
                    "end": end,
                    "step": step,
                    "results": [r.to_dict() for r in results]
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/read")
        async def read_metric(
            metric_name: str,
            labels: str | None = None,
            start: int | None = None,
            end: int | None = None,
            limit: int = 10000,
        ):
            """Read metric data directly."""
            try:
                label_dict = json.loads(labels) if labels else None
                results = self.db.read(
                    metric_name=metric_name,
                    labels=label_dict,
                    start=start,
                    end=end,
                    limit=limit,
                )
                return {
                    "count": len(results),
                    "series": [
                        {
                            "metric": ts.metric.to_prometheus_format(),
                            "points": [(p.timestamp, p.value) for p in ts.points]
                        }
                        for ts in results
                    ]
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/series")
        async def series(match: str | None = None):
            """List all series."""
            try:
                results = self.db.series(match)
                return {"count": len(results), "series": results}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/label_values")
        async def label_values(label_name: str):
            """Get unique label values."""
            try:
                values = self.db.label_values(label_name)
                return {"label": label_name, "values": list(values)}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/vector/search")
        async def search_vectors(
            vector: list[float],
            k: int = 10,
            metric_name: str | None = None,
            time_start: int | None = None,
            time_end: int | None = None,
        ):
            """Search for similar vectors."""
            try:
                import numpy as np
                time_range = None
                if time_start and time_end:
                    time_range = (time_start, time_end)
                
                results = self.db.search_vectors(
                    query_vector=np.array(vector),
                    k=k,
                    metric_name=metric_name,
                    time_range=time_range,
                )
                return {
                    "results": [
                        {
                            "id": r.id,
                            "distance": r.distance,
                            "score": r.score,
                            "metric_name": r.metric_name,
                            "labels": r.labels,
                        }
                        for r in results
                    ]
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/vector/add")
        async def add_vector(data: dict[str, Any]):
            """Add a vector embedding."""
            try:
                import numpy as np
                self.db.add_vector(
                    metric_name=data["metric_name"],
                    labels=data.get("labels", {}),
                    vector=np.array(data["vector"]),
                    timestamp=data.get("timestamp"),
                    value=data.get("value"),
                )
                return {"success": True}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.delete("/drop")
        async def drop_metric(
            metric_name: str,
            labels: str | None = None,
        ):
            """Delete metrics."""
            try:
                label_dict = json.loads(labels) if labels else None
                count = self.db.drop_metric(metric_name, label_dict)
                return {"success": True, "deleted": count}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/export")
        async def export(format: str = "json"):
            """Export data."""
            try:
                data = self.db.export(format)
                return Response(
                    content=data,
                    media_type="application/json" if format == "json" else "text/plain"
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
    
    def run(self, host: str | None = None, port: int | None = None) -> None:
        """Run the API server."""
        if not FASTAPI_AVAILABLE:
            print("Error: FastAPI required for API server")
            return
        
        import uvicorn
        
        host = host or self.config.host
        port = port or self.config.port
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info" if self.config.debug else "warning",
        )


def create_app(database: Any) -> Any:
    """
    Create FastAPI application.
    
    Args:
        database: PrometheusDB instance
    
    Returns:
        FastAPI application
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI required: pip install fastapi uvicorn")
    
    server = APIServer(database)
    return server.app
