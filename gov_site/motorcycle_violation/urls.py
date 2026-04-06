from django.urls import path
from . import views  # Import all views

urlpatterns = [
    path('', views.landing_page, name='landing_page'),
    
    # User account management
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('change_password/', views.change_password, name='change_password'),
    
    # Citizen dashboard
    path('citizen_dashboard/', views.citizen_dashboard, name='citizen_dashboard'),

    # Officer dashboard
    path('officer_dashboard/', views.officer_dashboard, name='officer_dashboard'),
    path('officer_profile/', views.officer_profile, name='officer_profile'),  # URL for officer's profile page

    # Reporting violations
    path('report-violation/', views.report_violation, name='report_violation'),  # URL for reporting a violation
    path('report-history/', views.report_history, name='report_history'),  # URL for viewing report history

    # Motorbike management
    path('manage_motorbikes/', views.manage_motorbikes, name='manage_motorbikes'),
    path('see_all_motorbikes/', views.see_all_motorbikes, name='see_all_motorbikes'),  # URL for seeing all motorbikes

    # Appeals management
    path('review_appeals/', views.review_appeals, name='review_appeals'),  # URL for reviewing appeals
    path('reviewed_appeal_history/', views.reviewed_appeal_history, name='reviewed_appeal_history'),  # URL for reviewed appeals history
    path('appeal_history/', views.appeal_history, name='appeal_history'),

    # Other functionalities
    path('process-license-plate/', views.process_license_plate, name='process_license_plate'),
    path('fine_history/', views.fine_history, name='fine_history'),  # URL for fine history
    path('download-receipt/', views.download_receipt, name='download_receipt'),
    path('download-receipt/<uuid:receipt_id>/', views.download_receipt_pdf, name='download_receipt_pdf'),
    path('receipt_history/', views.receipt_history, name='receipt_history'),

    path('file_appeal/', views.file_appeal, name='file_appeal'),

    path('splash/', views.splash_screen, name='splash_screen'),

    path('verify_receipt/<str:receipt_id>/', views.verify_receipt, name='verify_receipt'),

    # New URL for video upload
    path('upload-video/', views.upload_video, name='upload_video'),

    path('clear_media_redirect/', views.clear_media_redirect, name='clear_media_redirect'),
    path('report-license-plate/', views.report_license_plate, name='report_license_plate'),

    path('receipts/', views.receipt_list, name='receipt_list'),
    path('receipts/update/<int:receipt_id>/', views.update_receipt_status, name='update_receipt_status'),

    path('suspensions/overview/', views.suspension_overview, name='suspension_overview'),
    path('suspensions/history/', views.suspension_history, name='suspension_history'),

    path('chatbot-api/', views.chatbot_api, name='chatbot_api'),
]
