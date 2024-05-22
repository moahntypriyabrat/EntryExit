from django.shortcuts import render
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .forms import VideoForm
from .models import Video
from .task import detect_object


def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
            detect_object(video.id)
            return redirect('video_detail', video.id)
    else:
        form = VideoForm()
    return render(request, 'count/upload.html', {'form': form})


def video_detail(request, pk):
    video = Video.objects.get(pk=pk)
    return render(request, 'count/details.html', {'video': video})

