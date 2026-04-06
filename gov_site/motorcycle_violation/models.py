# === Standard Library ===
import math
import os
from datetime import timedelta

# === Django Core ===
from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.core.validators import RegexValidator
from django.db import models
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.utils import timezone

# === Project Utilities ===
from .utils import create_license_plate_bike_front

class CustomUser(AbstractUser):
    USER_TYPE_CHOICES = [
        ('citizen', 'Citizen'),
        ('officer', 'Government Officer'),
    ]
    
    user_type = models.CharField(max_length=10, choices=USER_TYPE_CHOICES)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    email = models.EmailField(unique=True)

    def __str__(self):
        return self.username

class CitizenProfile(models.Model):
    user = models.OneToOneField('CustomUser', on_delete=models.CASCADE, related_name='citizen_profile')
    cnic = models.CharField(max_length=15, blank=True, null=True, unique=True)
    credit_balance = models.DecimalField(default=0, max_digits=10, decimal_places=2)

    @property
    def is_currently_suspended(self):
        """
        Returns True if the citizen currently has any active suspension,
        i.e., there is at least one suspension where suspended_at <= now < suspended_until.
        """
        now = timezone.now()
        return self.suspension_history.filter(suspended_at__lte=now, suspended_until__gt=now).exists()


    @property
    def total_suspension_duration_left(self):
        """
        Returns the total suspension time left (in days) for the citizen by taking the
        active suspension with the latest suspended_until datetime, computing the time
        difference from now, and rounding up any partial day. If there are no active suspensions,
        or if the latest suspended_until is not in the future, return 0.
        """
        now = timezone.now()
        active_suspensions = self.suspension_history.filter(suspended_until__gt=now)
        if active_suspensions.exists():
            max_suspended_until = active_suspensions.order_by('-suspended_until').first().suspended_until
            if max_suspended_until <= now:
                return 0
            remaining = max_suspended_until - now
            return math.ceil(remaining.total_seconds() / 86400)
        return 0

    @property
    def total_suspension_fines(self):
        """
        Returns the total suspension fines for the citizen by summing the fine field for
        each active suspension that has not been paid.
        """
        now = timezone.now()
        active_suspensions = self.suspension_history.filter(suspended_until__gt=now, is_paid=False)
        total_fines = sum(s.fine for s in active_suspensions)
        return total_fines


    def __str__(self):
        return f"Citizen: {self.user.username} - CNIC: {self.cnic}"


class OfficerProfile(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='officer_profile')
    employee_id = models.CharField(max_length=10, blank=True, null=True, unique=True)

    def __str__(self):
        return f"Officer: {self.user.username} - Employee ID: {self.employee_id}"


# Signal to create profiles automatically when a CustomUser is created
@receiver(post_save, sender=CustomUser)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        if instance.user_type == 'citizen':
            CitizenProfile.objects.create(user=instance)
        elif instance.user_type == 'officer' or instance.is_superuser:
            OfficerProfile.objects.create(user=instance)

# Remove profile creation logic from signals
@receiver(post_save, sender=CustomUser)
def save_user_profile(sender, instance, **kwargs):
    if instance.user_type == 'citizen':
        if hasattr(instance, 'citizen_profile'):
            instance.citizen_profile.save()  # Only call save if the profile exists
    elif instance.user_type == 'officer':
        if hasattr(instance, 'officer_profile'):
            instance.officer_profile.save()  # Only call save if the profile exists


class Motorcycle(models.Model):
    # Regex to validate registration number
    registration_number_validator = RegexValidator(
        regex=r'^[A-Z]{3}-\d{3,4}(-\d{2})?(-[A-Z])?$',
        message='Registration number must be in the format: ABC-1234-12-A'
    )

    owner = models.ForeignKey(CitizenProfile, on_delete=models.CASCADE, related_name='motorcycles')
    registration_number = models.CharField(
        max_length=20,
        unique=True,
        validators=[registration_number_validator]  # Apply the validator
    )
    model_name = models.CharField(max_length=50)
    fine_balance = models.DecimalField(default=0, max_digits=10, decimal_places=2)

    def __str__(self):
        return f"{self.registration_number} ({self.model_name})"

    @property
    def plate_image_path(self):
        """Generate the file path for the license plate image."""
        ler, number, year, single = self.parse_registration_number()
        file_name = f"{ler}{number}{year}{single}.jpg"
        return os.path.join(settings.MEDIA_URL, "generated_license_plates", file_name)

    def parse_registration_number(self):
        """Parse the registration number into its components."""
        parts = self.registration_number.split('-')
        ler = parts[0]
        number = parts[1]
        year = parts[2] if len(parts) > 2 else ""
        single = parts[3] if len(parts) > 3 else ""
        return ler, number, year, single


@receiver(post_save, sender=Motorcycle)
def create_license_plate_on_save(sender, instance, created, **kwargs):
    """Create a license plate image when a Motorcycle is added."""
    if created:
        ler, number, year, single = instance.parse_registration_number()
        create_license_plate_bike_front(
            ler_text=ler,
            number_text=number,
            manufacture_year=year,
            single_text=single
        )


@receiver(post_delete, sender=Motorcycle)
def delete_license_plate_on_delete(sender, instance, **kwargs):
    """Delete the license plate image when a Motorcycle is removed."""
    plate_path = instance.plate_image_path
    if os.path.exists(plate_path):
        os.remove(plate_path)



class Violation(models.Model):
    REPORT_STATUS_CHOICES = [
        ('no', 'No Appeal'),
        ('yes', 'Pending'),
        ('accepted', 'Accepted'),
        ('rejected', 'Rejected'),
    ]

    reporter = models.ForeignKey(CitizenProfile, on_delete=models.CASCADE, related_name='violations')
    motorcycle = models.ForeignKey(Motorcycle, on_delete=models.CASCADE)
    violation_type = models.CharField(max_length=100)
    license_plate = models.CharField(max_length=20)
    timestamp = models.DateTimeField(auto_now_add=True)
    is_downloaded = models.BooleanField(default=False)
    cropped_image = models.ImageField(upload_to='cropped_images/', blank=True, null=True)
    is_appealed = models.CharField(max_length=10, choices=REPORT_STATUS_CHOICES, default='no')

    # Store per-violation fixed values
    fine_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    credit_rewarded = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    def save(self, *args, **kwargs):
        if not self.pk:
            # Freeze values at creation
            self.fine_amount = settings.VIOLATION_FINE_AMOUNT
            self.credit_rewarded = settings.VIOLATION_CREDIT_REWARD

            # Reward credits to the reporter
            self.reporter.credit_balance += self.credit_rewarded
            self.reporter.save()

            # Update motorcycle fine balance
            self.motorcycle.fine_balance += self.fine_amount
            self.motorcycle.save()

        super().save(*args, **kwargs)

    def __str__(self):
        return f"Violation: {self.violation_type} - {self.license_plate}"


class Receipt(models.Model):
    user = models.ForeignKey('CitizenProfile', on_delete=models.CASCADE)
    current_fines = models.DecimalField(max_digits=10, decimal_places=2)
    current_credits = models.DecimalField(max_digits=10, decimal_places=2)
    suspension_fines = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    payable_amount = models.DecimalField(max_digits=10, decimal_places=2)
    receipt_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # New field to store the timestamp when the status is updated from GENERATED
    status_updated_at = models.DateTimeField(null=True, blank=True)

    STATUS_CHOICES = (
        ('PAID', 'Paid'),
        ('REDEEMED', 'Redeemed'),
        ('GENERATED', 'Generated'),
    )

    status = models.CharField(
        max_length=10,
        choices=STATUS_CHOICES,
        default='GENERATED'
    )

    def save(self, *args, **kwargs):
        # Only check for changes if the instance already exists
        if self.pk:
            original = Receipt.objects.get(pk=self.pk)
            # If the original status is 'GENERATED' and it's changing to 'PAID' or 'REDEEMED'
            if (original.status == 'GENERATED' and 
                self.status in ['PAID', 'REDEEMED'] and 
                not self.status_updated_at):
                self.status_updated_at = timezone.now()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Receipt {self.receipt_id} for {self.user.user.username}"


class MotorcycleReceipt(models.Model):
    receipt = models.ForeignKey(Receipt, related_name='motorcycle_receipts', on_delete=models.CASCADE)
    motorcycle = models.ForeignKey(Motorcycle, on_delete=models.CASCADE)
    fine_balance = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return f"{self.motorcycle.registration_number} - Receipt {self.receipt.receipt_id} - Fine: {self.fine_balance}"
    
class Appeal(models.Model):
    citizen = models.ForeignKey(CitizenProfile, on_delete=models.CASCADE, related_name='appeals')
    violation = models.ForeignKey(Violation, on_delete=models.CASCADE, related_name='appeals')
    officer = models.ForeignKey(OfficerProfile, on_delete=models.SET_NULL, null=True, blank=True, related_name='handled_appeals')
    reason = models.TextField()
    status = models.CharField(max_length=10, choices=[
        ('pending', 'Pending'),
        ('accepted', 'Accepted'),
        ('rejected', 'Rejected'),
    ], default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Appeal by {self.citizen.user.username} for {self.violation.violation_type} - Status: {self.status}"


class Suspension(models.Model):
    citizen = models.ForeignKey(
        'CitizenProfile',
        on_delete=models.CASCADE,
        related_name='suspension_history'
    )
    violation = models.OneToOneField(
        'Violation',
        on_delete=models.CASCADE,
        related_name='suspension'
    )
    reason = models.TextField(blank=True, null=True)
    suspended_by = models.ForeignKey(
        'OfficerProfile',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='issued_suspensions'
    )
    suspended_at = models.DateTimeField(blank=True, null=True)
    suspended_until = models.DateTimeField(blank=True, null=True)
    fine = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    is_paid = models.BooleanField(default=False)  # New field for payment status

    def save(self, *args, **kwargs):
        """
        When a new suspension is added, recalculate the suspension period as follows:
          - Determine the provided duration (in days) and round it up.
          - Look up all active suspensions (those with suspended_until > now).
          - Determine the new starting point (suspended_at) as the later of current time or
            the largest suspended_until from active suspensions.
          - Compute suspended_until as suspended_at + provided_duration.
        """
        # Pop the custom provided_duration kwarg (default to 3 days if not provided)
        provided_duration = kwargs.pop('provided_duration', None)
        try:
            # Allow fractional days (e.g., 1.5) and round up to the next whole day.
            duration_days = float(provided_duration) if provided_duration is not None else 3
        except (ValueError, TypeError):
            duration_days = 3
        duration_days = math.ceil(duration_days)

        current_time = timezone.now()

        # Look up all active suspensions (those ending after now)
        active_suspensions = self.citizen.suspension_history.filter(suspended_until__gt=current_time)
        if active_suspensions.exists():
            # Get the latest suspended_until datetime among them.
            max_suspended_until = active_suspensions.order_by('-suspended_until').first().suspended_until
            new_suspended_at = max_suspended_until if max_suspended_until > current_time else current_time
        else:
            new_suspended_at = current_time

        # Always override suspended_at and suspended_until with our computed values.
        self.suspended_at = new_suspended_at
        self.suspended_until = self.suspended_at + timedelta(days=duration_days)
        
        super().save(*args, **kwargs)

    @property
    def duration(self):
        """
        Returns the duration (in days) of this suspension, rounding up any partial day.
        """
        if not self.suspended_at or not self.suspended_until:
            return 0
        delta = self.suspended_until - self.suspended_at
        return math.ceil(delta.total_seconds() / 86400)

    @property
    def is_active(self):
        now = timezone.now()
        if not self.suspended_at or not self.suspended_until:
            return False
        return self.suspended_at <= now < self.suspended_until

    @property
    def is_complete(self):
        now = timezone.now()
        if not self.suspended_until:
            return False
        return now >= self.suspended_until

    def __str__(self):
        suspended_by_str = self.suspended_by.user.username if self.suspended_by else "Unknown"
        end_str = self.suspended_until.strftime('%Y-%m-%d %H:%M:%S') if self.suspended_until else "Unknown"
        return f"Suspension for {self.citizen.user.username} by {suspended_by_str} (Active until: {end_str})"
