from rest_framework import serializers
from .models import ScheduleRequest, ScheduleGrade

class ScheduleRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = ScheduleRequest
        fields = '__all__'


class ScheduleGradeSerializer(serializers.ModelSerializer):
    class Meta:
        model = ScheduleGrade
        fields = ['schedule_request', 'is_approved']  # No 'score'

