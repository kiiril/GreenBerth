from django.http import JsonResponse
from django.urls import path
from .views import ScheduleRequestCreateView, ScheduleEvaluationCreateView


def root_view(request):
    return JsonResponse({"message": "Scheduler API is running"})

urlpatterns = [
    path('', root_view),
    path('api/v1/schedules', ScheduleRequestCreateView.as_view(), name='create_schedule'),
    path('api/v1/schedules/<uuid:pk>/evaluations', ScheduleEvaluationCreateView.as_view(), name='evaluate_schedule'),
]
