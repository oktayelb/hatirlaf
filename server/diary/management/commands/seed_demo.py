"""Demo-seed command.

Kept as a no-op safety net: we used to auto-seed sample entries so the
UI had something to show. Real users don't want mock sessions leaking
into their diary, so the command now refuses to run unless explicitly
asked with ``--force``.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from diary.models import Session


class Command(BaseCommand):
    help = "Demo data is disabled by default. Use --wipe to clear any existing sessions."

    def add_arguments(self, parser):
        parser.add_argument(
            "--wipe",
            action="store_true",
            help="Delete every session row (use to clear previously seeded demo data).",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="(deprecated) Re-run the old demo seed. Not recommended.",
        )

    def handle(self, *args, **opts):
        if opts.get("wipe"):
            n = Session.objects.count()
            Session.objects.all().delete()
            self.stdout.write(self.style.SUCCESS(f"Removed {n} session(s)."))
            return

        self.stdout.write(
            "Demo seeding is disabled. Use '--wipe' to clear any existing sessions."
        )
