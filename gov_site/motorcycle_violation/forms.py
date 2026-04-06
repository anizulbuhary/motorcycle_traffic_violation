from .models import CustomUser, CitizenProfile, OfficerProfile, Motorcycle
from django.core.exceptions import ValidationError
from django import forms
import re

class CitizenUserCreationForm(forms.ModelForm):
    cnic = forms.CharField(required=True)

    def clean_username(self):
        username = self.cleaned_data.get('username')

        # Enforce minimum length
        if len(username) < 5:
            raise ValidationError("Username must be at least 5 characters long.")

        # Only allow alphanumeric usernames (optionally allow _ or .)
        if not re.match(r'^[a-zA-Z0-9_.]+$', username):
            raise ValidationError("Username can only contain letters, numbers, underscores, or dots.")

        # Disallow common weak usernames
        weak_usernames = {'admin', 'user', 'test', 'guest', 'username'}
        if username.lower() in weak_usernames:
            raise ValidationError("This username is too common. Please choose a more unique one.")

        # Check uniqueness (if not already enforced by unique=True on the model)
        if CustomUser.objects.filter(username=username).exists():
            raise ValidationError("This username is already taken.")

        return username


    class Meta:
        model = CustomUser
        fields = ('first_name', 'last_name', 'username', 'email', 'password', 'cnic')
        widgets = {
            'password': forms.PasswordInput(),
        }

    def clean_first_name(self):
        first_name = self.cleaned_data.get('first_name')
        if not re.match(r'^[A-Za-z\s]+$', first_name):
            raise ValidationError("First name should only contain letters and spaces.")
        return first_name

    def clean_last_name(self):
        last_name = self.cleaned_data.get('last_name')
        if not re.match(r'^[A-Za-z\s]+$', last_name):
            raise ValidationError("Last name should only contain letters and spaces.")
        return last_name

    def clean_password(self):
        password = self.cleaned_data.get('password')
        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters long.")
        if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
            raise ValidationError("Password must contain both letters and numbers.")
        return password

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if CustomUser.objects.filter(email=email).exists():
            raise ValidationError("An account with this email already exists.")
        return email

    def clean_cnic(self):
        cnic = self.cleaned_data.get('cnic')
        if CitizenProfile.objects.filter(cnic=cnic).exists():
            raise ValidationError("An account with this CNIC already exists.")
        return cnic

    def clean(self):
        cleaned_data = super().clean()
        cnic = cleaned_data.get('cnic')

        if not cnic or len(cnic) != 13 or not cnic.isdigit():
            raise ValidationError("CNIC must be exactly 13 digits.")

        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data['password'])
        user.user_type = 'citizen'
        if commit:
            user.save()
            CitizenProfile.objects.update_or_create(user=user, defaults={'cnic': self.cleaned_data['cnic']})
        return user



class OfficerUserCreationForm(forms.ModelForm):
    employee_id = forms.CharField(required=True)

    class Meta:
        model = CustomUser
        fields = ('first_name', 'last_name', 'username', 'email', 'password', 'employee_id')
        widgets = {
            'password': forms.PasswordInput(),
        }

    def clean_username(self):
        username = self.cleaned_data.get('username')

        # Enforce minimum length
        if len(username) < 5:
            raise ValidationError("Username must be at least 5 characters long.")

        # Only allow alphanumeric usernames (optionally allow _ or .)
        if not re.match(r'^[a-zA-Z0-9_.]+$', username):
            raise ValidationError("Username can only contain letters, numbers, underscores, or dots.")

        # Disallow common weak usernames
        weak_usernames = {'admin', 'user', 'test', 'guest', 'username'}
        if username.lower() in weak_usernames:
            raise ValidationError("This username is too common. Please choose a more unique one.")

        # Check uniqueness (if not already enforced by unique=True on the model)
        if CustomUser.objects.filter(username=username).exists():
            raise ValidationError("This username is already taken.")

        return username
    
    def clean_first_name(self):
        first_name = self.cleaned_data.get('first_name')
        if not re.match(r'^[A-Za-z\s]+$', first_name):
            raise ValidationError("First name should only contain letters and spaces.")
        return first_name

    def clean_last_name(self):
        last_name = self.cleaned_data.get('last_name')
        if not re.match(r'^[A-Za-z\s]+$', last_name):
            raise ValidationError("Last name should only contain letters and spaces.")
        return last_name

    def clean_password(self):
        password = self.cleaned_data.get('password')
        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters long.")
        if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
            raise ValidationError("Password must contain both letters and numbers.")
        return password

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if CustomUser.objects.filter(email=email).exists():
            raise ValidationError("An account with this email already exists.")
        return email

    def clean_employee_id(self):
        employee_id = self.cleaned_data.get('employee_id')
        if OfficerProfile.objects.filter(employee_id=employee_id).exists():
            raise ValidationError("An account with this Employee ID already exists.")
        return employee_id

    def clean(self):
        cleaned_data = super().clean()
        employee_id = cleaned_data.get('employee_id')

        if not employee_id or len(employee_id) != 10 or not employee_id.isdigit():
            raise ValidationError("Employee ID must be exactly 10 digits.")

        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data['password'])
        user.user_type = 'officer'
        if commit:
            user.save()
            OfficerProfile.objects.update_or_create(user=user, defaults={'employee_id': self.cleaned_data['employee_id']})
        return user


class ReportViolationForm(forms.Form):
    image = forms.ImageField(label='Upload Image', required=True)
    license_plate = forms.CharField(label='License Plate Number', max_length=15, required=False, help_text="Optional")


class MotorcycleForm(forms.ModelForm):
    class Meta:
        model = Motorcycle
        fields = ['registration_number', 'model_name']  
