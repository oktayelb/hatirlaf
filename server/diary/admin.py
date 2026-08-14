from django.contrib import admin

from .models import Edge, Mention, Node, Session


@admin.register(Node)
class NodeAdmin(admin.ModelAdmin):
    list_display = ("label", "kind", "is_unknown", "updated_at")
    list_filter = ("kind", "is_unknown")
    search_fields = ("label", "aliases", "notes")


@admin.register(Session)
class SessionAdmin(admin.ModelAdmin):
    list_display = ("client_uuid", "recorded_at", "status", "duration_seconds")
    list_filter = ("status",)
    search_fields = ("client_uuid", "transcript")


@admin.register(Mention)
class MentionAdmin(admin.ModelAdmin):
    list_display = ("surface", "mention_type", "session", "node", "is_conflict", "resolved")
    list_filter = ("mention_type", "is_conflict", "resolved", "conflict_reason")
    search_fields = ("surface", "lemma")


@admin.register(Edge)
class EdgeAdmin(admin.ModelAdmin):
    list_display = ("source", "relation", "target", "session", "created_at")
    list_filter = ("relation",)
