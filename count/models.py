from django.db import models


class Video(models.Model):
    title = models.CharField(max_length=100)
    original_video = models.FileField(upload_to='videos/original/')
    processed_video = models.FileField(upload_to='videos/processed/', null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
