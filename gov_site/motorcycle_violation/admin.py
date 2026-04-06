from django.contrib import admin
from .models import CustomUser, CitizenProfile, OfficerProfile, Motorcycle, Violation, Receipt, Appeal, MotorcycleReceipt, Suspension
from django.contrib.auth.admin import UserAdmin
from django.utils.html import format_html

admin.site.site_header = "SafeRide Administration"
admin.site.site_title = "SafeRide Admin Portal"
admin.site.index_title = "Welcome to SafeRide Admin Dashboard"

# Register the CustomUser model with the default UserAdmin interface
@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'first_name', 'last_name', 'user_type', 'is_staff', 'is_active')
    list_filter = ('user_type', 'is_staff', 'is_active', 'is_superuser')
    fieldsets = UserAdmin.fieldsets + (
        (None, {'fields': ('user_type',)}),
    )

# Register CitizenProfile model
@admin.register(CitizenProfile)
class CitizenProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'cnic', 'credit_balance')
    search_fields = ('user__username', 'cnic')

# Register OfficerProfile model
@admin.register(OfficerProfile)
class OfficerProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'employee_id')
    search_fields = ('user__username', 'employee_id')

# Register Motorcycle model
@admin.register(Motorcycle)
class MotorcycleAdmin(admin.ModelAdmin):
    list_display = ('registration_number', 'model_name', 'owner', 'fine_balance')
    search_fields = ('registration_number', 'model_name', 'owner__user__username')
    list_filter = ('owner',)

@admin.register(Violation)
class ViolationAdmin(admin.ModelAdmin):
    list_display = (
        'violation_type',
        'license_plate',
        'motorcycle',
        'reporter',
        'timestamp',
        'is_appealed',
        'is_downloaded',
        'fine_amount',
        'credit_rewarded',
        'show_cropped_image',
    )
    search_fields = (
        'violation_type',
        'license_plate',
        'reporter__user__username',
    )
    list_filter = (
        'is_downloaded',
        'timestamp',
        'is_appealed',
    )

    @admin.display(description='Cropped Image')
    def show_cropped_image(self, obj):
        if obj.cropped_image:
            return format_html('<img src="{}" style="width: 50px; height: auto;" />', obj.cropped_image.url)
        return "No Image"

# Register Receipt model
@admin.register(Receipt)
class ReceiptAdmin(admin.ModelAdmin):
    list_display = ('receipt_id', 'user', 'status', 'current_fines', 'suspension_fines', 'current_credits', 'payable_amount', 'created_at')
    search_fields = ('receipt_id', 'user__user__username', 'status')
    list_filter = ('created_at', 'status')

# Register MotorcycleReceipt model
@admin.register(MotorcycleReceipt)
class MotorcycleReceiptAdmin(admin.ModelAdmin):
    list_display = ('motorcycle', 'receipt', 'fine_balance')
    search_fields = ('motorcycle__registration_number', 'receipt__receipt_id')
    list_filter = ('receipt__created_at',)

# Register Appeal model
@admin.register(Appeal)
class AppealAdmin(admin.ModelAdmin):
    list_display = ('id', 'citizen', 'violation', 'officer', 'status', 'created_at', 'updated_at')
    search_fields = ('citizen__user__username', 'violation__violation_type', 'reason')
    list_filter = ('status', 'created_at', 'officer')
    ordering = ('-created_at',)  # Orders appeals by creation date (newest first)

    def get_queryset(self, request):
        # Optimize the queryset for related fields
        return super().get_queryset(request).select_related('citizen', 'violation', 'officer')


@admin.register(Suspension)
class SuspensionAdmin(admin.ModelAdmin):
    list_display = (
        'citizen', 
        'violation',
        'fine', 
        'is_paid',
        'suspended_by', 
        'suspended_at', 
        'suspended_until', 
        'duration', 
        'is_active'
    )
    list_filter = ('suspended_by', 'citizen')
    search_fields = (
        'citizen__user__username',  # assumes the related CustomUser has a 'username'
        'violation__id',
        'reason'
    )
    ordering = ('-suspended_until',)
