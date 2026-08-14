from django.shortcuts import render


def index(request):
    """Serve the mobile-style web client shell."""
    return render(request, "diary/index.html")
