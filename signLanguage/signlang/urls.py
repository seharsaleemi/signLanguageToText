from django.urls import path

from . import views

urlpatterns = [
    path('', views.Home, name='index'),
    path('test_stream', views.test_stream, name='test_stream'),
    path('text_stream', views.text_stream, name='text_stream'),
]
