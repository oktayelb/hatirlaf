from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

from diary.views import web

urlpatterns = [
    path("", web.index, name="web-index"),
    path("admin/", admin.site.urls),
    path("api/", include("diary.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.BASE_DIR / "static")
