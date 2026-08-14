from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("diary", "0004_encounteredentity"),
    ]

    operations = [
        migrations.CreateModel(
            name="PrivacySettings",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("password_hash", models.CharField(blank=True, default="", max_length=256)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name": "Privacy settings",
                "verbose_name_plural": "Privacy settings",
            },
        ),
    ]
