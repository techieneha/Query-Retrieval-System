from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict
import statistics
import logging

logger = logging.getLogger(__name__)


class AnalyticsTracker:
    """Track query performance and patterns"""
    
    def __init__(self):
        self.queries = []
        self.max_history = 1000
        
    def log_query(self, data: dict):
        """Log query with metadata"""
        self.queries.append({
            **data,
            'timestamp': datetime.now()
        })
        
        if len(self.queries) > self.max_history:
            self.queries = self.queries[-self.max_history:]
    
    def get_stats(self, hours: int = 24) -> dict:
        """Get analytics for past N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [q for q in self.queries if q['timestamp'] > cutoff]
        
        if not recent:
            return {
                'total_queries': 0,
                'time_range_hours': hours,
                'message': 'No queries in this time period'
            }
        
        latencies = [q['latency'] for q in recent]
        confidences = [q['confidence'] for q in recent]
        cached = sum(1 for q in recent if q.get('cached', False))
        
        quality_counts = defaultdict(int)
        for q in recent:
            quality_counts[q.get('quality', 'unknown')] += 1
        
        latencies_sorted = sorted(latencies)
        
        def percentile(data, p):
            idx = int(len(data) * p)
            return data[min(idx, len(data) - 1)]
        
        return {
            'total_queries': len(recent),
            'time_range_hours': hours,
            'latency': {
                'mean': round(statistics.mean(latencies), 3),
                'median': round(statistics.median(latencies), 3),
                'p95': round(percentile(latencies_sorted, 0.95), 3),
                'p99': round(percentile(latencies_sorted, 0.99), 3),
                'min': round(min(latencies), 3),
                'max': round(max(latencies), 3)
            },
            'confidence': {
                'mean': round(statistics.mean(confidences), 3),
                'median': round(statistics.median(confidences), 3),
                'min': round(min(confidences), 3),
                'max': round(max(confidences), 3)
            },
            'cache': {
                'hits': cached,
                'misses': len(recent) - cached,
                'hit_rate': round(cached / len(recent) * 100, 2)
            },
            'quality_distribution': dict(quality_counts),
            'queries_per_hour': round(len(recent) / hours, 2)
        }
    
    def get_popular_queries(self, limit: int = 10) -> List[Dict]:
        """Get most common queries"""
        query_counts = defaultdict(int)
        
        for q in self.queries:
            normalized = q['query'].lower().strip()
            query_counts[normalized] += 1
        
        sorted_queries = sorted(
            query_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {'query': query, 'count': count}
            for query, count in sorted_queries
        ]


analytics_tracker = AnalyticsTracker()