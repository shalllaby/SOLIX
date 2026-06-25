import time
import os
import sys
from typing import Dict, List
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

class SimpleRateLimiter(BaseHTTPMiddleware):
    def __init__(self, app, requests_limit: int = 30, window_seconds: int = 60):
        """
        Simple memory-based rate limiter middleware.
        Defaults: Max 30 requests per 60 seconds per IP address.
        """
        super().__init__(app)
        self.limit = requests_limit
        self.window = window_seconds
        self.history: Dict[str, List[float]] = {}

    async def dispatch(self, request: Request, call_next):
        # We only rate limit auth API endpoints to avoid rate-limiting static/frontend page loads
        if not request.url.path.startswith("/api/v1/auth"):
            return await call_next(request)

        client_ip = request.headers.get("x-test-ip") or (request.client.host if request.client else "unknown")
        
        # Bypass rate limiter in testing environment, unless specifically testing rate limit
        if ("pytest" in sys.modules or os.getenv("TESTING") == "1") and client_ip != "ratelimit-ip":
            return await call_next(request)

        now = time.time()

        # Clean up history for this client
        if client_ip in self.history:
            self.history[client_ip] = [
                t for t in self.history[client_ip] if now - t < self.window
            ]
        else:
            self.history[client_ip] = []

        # Check rate limit
        if len(self.history[client_ip]) >= self.limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Rate limit exceeded. Please wait before retrying."}
            )

        # Track this request
        self.history[client_ip].append(now)
        
        response = await call_next(request)
        return response
