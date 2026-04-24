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
    timeline_view,
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
]
