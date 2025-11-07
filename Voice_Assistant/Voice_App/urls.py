from django.urls import path
from .views import index, api_ask, reset_context

urlpatterns = [
    path('', index, name='voice_index'),
    path('ask/', api_ask, name='api_ask'),
    path('reset/', reset_context, name='reset_context'),
]