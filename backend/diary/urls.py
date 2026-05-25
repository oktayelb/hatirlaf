from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views.api import (
    EdgeListView,
    MentionViewSet,
    NodeViewSet,
    SessionViewSet,
    calendar_view,
    graph_view,
    health_view,
    recap_view,
    timeline_view,
)
from .views.api_privacy import (
    privacy_clear_password_view,
    privacy_lock_view,
    privacy_set_password_view,
    privacy_status_view,
    privacy_unlock_view,
)

router = DefaultRouter()
router.register(r"sessions", SessionViewSet, basename="session")
router.register(r"mentions", MentionViewSet, basename="mention")
router.register(r"nodes", NodeViewSet, basename="node")

urlpatterns = [
    path("", include(router.urls)),
    path("edges/", EdgeListView.as_view(), name="edge-list"),
    path("timeline/", timeline_view, name="timeline"),
    path("calendar/", calendar_view, name="calendar"),
    path("graph/", graph_view, name="graph"),
    path("health/", health_view, name="health"),
    path("recap/", recap_view, name="recap"),
    path("privacy/status/", privacy_status_view, name="privacy-status"),
    path("privacy/unlock/", privacy_unlock_view, name="privacy-unlock"),
    path("privacy/lock/", privacy_lock_view, name="privacy-lock"),
    path("privacy/set-password/", privacy_set_password_view, name="privacy-set-password"),
    path("privacy/clear-password/", privacy_clear_password_view, name="privacy-clear-password"),
]
