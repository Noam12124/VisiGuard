from api.auth import router as auth_router
from api.events import router as events_router
from api.gallery import router as gallery_router
from api.monitor import router as monitor_router
from api.unknowns import router as unknowns_router

__all__ = [
    "auth_router",
    "events_router",
    "gallery_router",
    "monitor_router",
    "unknowns_router",
]
