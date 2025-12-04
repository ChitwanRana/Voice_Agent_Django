from django.urls import path                                #type: ignore
from .views import index, api_ask, reset_context,api_tts,generate_avatar_audio  #type: ignore

urlpatterns = [
    path('', index, name='voice_index'),
    path('ask/', api_ask, name='api_ask'),
    path('reset/', reset_context, name='reset_context'),
    path('api/tts/',api_tts, name='api_tts'),
    path("generate_avatar_audio", generate_avatar_audio, name="generate_avatar_audio")

]