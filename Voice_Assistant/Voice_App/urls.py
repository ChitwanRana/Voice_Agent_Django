from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="voice_index"),
    path("ask/", views.api_ask, name="voice_api_ask"),
]