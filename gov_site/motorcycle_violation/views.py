# === Standard Library ===
import base64
import json
import shutil
import tempfile
import time
import uuid
from io import BytesIO

# === Third-party Libraries ===
import cv2  # OpenCV
import qrcode
from weasyprint import HTML
from ultralytics import YOLO

# === Django Core ===
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import PasswordChangeForm
from django.core.exceptions import PermissionDenied
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils import timezone
from django.utils.timezone import now
from django.views.decorators.csrf import csrf_exempt  # Use with caution

# === Project Imports ===
from .forms import (
    OfficerUserCreationForm,
    CitizenUserCreationForm,
    ReportViolationForm,
    MotorcycleForm,
)
from .models import (
    Motorcycle,
    CitizenProfile,
    OfficerProfile,
    Violation,
    Receipt,
    Appeal,
    MotorcycleReceipt,
    Suspension,
)
from .track.pipeline import main_pipeline
from .utils import *


def landing_page(request):
    return render(request, 'motorcycle_violation/landing_page.html')  # Renders the landing page template

@login_required
def splash_screen(request):
    user_type = request.user.user_type
    return render(request, 'motorcycle_violation/splash_screen.html', {'user_type': user_type})

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)

            # Redirect based on user type
            if user.user_type == 'citizen':
                try:
                    profile = user.citizen_profile  # Ensure CitizenProfile is linked
                    return redirect('splash_screen')  # Ensure this URL is defined
                except CitizenProfile.DoesNotExist:
                    messages.error(request, 'No citizen profile found for this user.')
                    return render(request, 'motorcycle_violation/login.html')

            elif user.user_type == 'officer':
                try:
                    profile = user.officer_profile  # Ensure OfficerProfile is linked
                    return redirect('splash_screen')  # Ensure this URL is defined
                except OfficerProfile.DoesNotExist:
                    messages.error(request, 'No officer profile found for this user.')
                    return render(request, 'motorcycle_violation/login.html')

        else:
            # Invalid username or password
            messages.error(request, 'Invalid username or password.')
            return render(request, 'motorcycle_violation/login.html')

    # GET request, simply render the login page
    return render(request, 'motorcycle_violation/login.html')


def register(request):
    form = None  # Initialize form to None

    if request.method == 'POST':
        user_type = request.POST.get('user_type')

        # Select the appropriate form based on user type
        if user_type == 'citizen':
            form = CitizenUserCreationForm(request.POST)
        elif user_type == 'officer':
            form = OfficerUserCreationForm(request.POST)
        else:
            # Handle invalid user type
            messages.error(request, 'Invalid user type selected.')
            return redirect('register')  # Redirect to register page

        # Validate the form
        if form and form.is_valid():
            form.save()
            messages.success(request, 'Your account has been created successfully! You can now log in.')
            return redirect('login')  # Redirect to login after successful registration

    return render(request, 'motorcycle_violation/register.html', {'form': form})

@login_required
def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            user = form.save()  # Save the new password
            messages.success(request, 'Your password was successfully updated!')
            return redirect('login')  # Redirect to the login page after successful change
        else:
            messages.error(request, 'Please correct the error below.')
            print(form.errors)  # Log form errors for debugging
    else:
        form = PasswordChangeForm(user=request.user)

    # Print user details for debugging
    print(f'User authenticated: {request.user.is_authenticated}')
    print(f'Current user: {request.user.username}')

    # Conditional rendering of the template based on user type
    if request.user.user_type == 'officer':
        template = 'motorcycle_violation/officer_change_password.html'  # Officer's change password page
    else:
        template = 'motorcycle_violation/change_password.html'  # Citizen's change password page

    # Passing the form and user details to the selected template
    return render(request, template, {
        'form': form,
        'current_user': request.user  # Passing user details to the template
    })


@login_required
def logout_view(request):
    logout(request)
    messages.success(request, 'User logged out successfully!')
    return redirect('login')  # Redirect to the login page after logout

@login_required
def profile_view(request):
    user = request.user  # Get the currently logged-in user
    
    # Initialize variables for credits
    credits = None

    # Check user type and retrieve relevant information
    if hasattr(user, 'citizen_profile'):
        # It's a citizen
        credits = user.citizen_profile.credit_balance

    return render(request, 'motorcycle_violation/profile.html', {
        'user': user,
        'credits': credits,
    })


@login_required
def officer_profile(request):
    user = request.user  # Get the currently logged-in user
    return render(request, 'motorcycle_violation/officer_profile.html', {
        'user': user,
    })

from django.http import HttpResponseForbidden

@login_required
def citizen_dashboard(request):
    user = request.user
    total_fine = 0
    total_suspension_fines = 0
    total_credits = 0
    payable_amount = 0  # Initialize payable amount

    if user.user_type != 'citizen':
        return HttpResponseForbidden("Access denied. Citizens only.")

    citizen_profile = user.citizen_profile
    total_fine = sum(motorcycle.fine_balance for motorcycle in citizen_profile.motorcycles.all())
    total_suspension_fines = citizen_profile.total_suspension_fines
    total_credits = citizen_profile.credit_balance or 0

    combined_fines = total_fine + total_suspension_fines
    payable_amount = max(combined_fines - total_credits, 0)

    context = {
        'total_fine': f"{total_fine:.2f}",
        'total_suspension_fines': f"{total_suspension_fines:.2f}",
        'total_credits': f"{total_credits:.2f}",
        'payable_amount': f"{payable_amount:.2f}",
    }

    return render(request, 'motorcycle_violation/citizen_dashboard.html', context)


@login_required
def officer_dashboard(request):
    user = request.user

    if user.user_type != 'officer':
        return HttpResponseForbidden("Access denied. Officers only.")

    officer_profile = user.officer_profile

    # Total number of appeals reviewed by this officer (Accepted or Rejected appeals)
    reviewed_appeals_count = Appeal.objects.filter(officer=officer_profile).exclude(status='pending').count()

    # Total number of pending appeals
    pending_appeals_count = Appeal.objects.filter(status='pending').count()

    # Total appeals handled today by this officer
    today = now().date()
    appeals_handled_today_count = Appeal.objects.filter(
        officer=officer_profile,
        updated_at__date=today
    ).exclude(status='pending').count()

    return render(request, 'motorcycle_violation/officer_dashboard.html', {
        'reviewed_appeals_count': reviewed_appeals_count,
        'pending_appeals_count': pending_appeals_count,
        'appeals_handled_today_count': appeals_handled_today_count,
    })



# # Configuration
# VIOLATION_API_URL       = "http://127.0.0.1:8001/detect/violation"
# MOTORCYCLE_API_URL      = "http://127.0.0.1:8001/detect/motorcycle"

# Helper functions
def get_api_detections(image_path: str, api_url: str):
    """Send image to specified API and return JSON response"""
    try:
        with open(image_path, 'rb') as img_file:
            response = requests.post(
                api_url,
                files={'file': img_file},
                timeout=30
            )
        if response.status_code == 200:
            return response.json()
        raise Exception(f"API error: {response.text}")
    except Exception as e:
        raise Exception(f"Failed to get detections: {str(e)}")


# Report violation view
@login_required
def report_violation(request):
    """
    Handles image uploads from citizens for reporting motorcycle violations.

    - Prevents reporting if user is suspended.
    - Accepts POST with image, runs detection for motorcycles and violations.
    - If valid, annotates image, extracts license plate text, and displays results.
    - On failure or invalid image, shows appropriate error message.

    Returns:
        Rendered template with form, result images, and extracted data.
    """

    if request.user.citizen_profile.is_currently_suspended:
        messages.error(request, "You are currently suspended and cannot report a violation.")
        return redirect('citizen_dashboard')

    error_message = None  # Initialize error message variable

    if request.method == 'POST':
        form = ReportViolationForm(request.POST, request.FILES)

        if 'image' in request.FILES and form.is_valid():
            image = request.FILES['image']

            # Sanitize and modify the image name before saving
            sanitized_name = get_valid_filename(image.name).replace(' ', '_')
            image.name = sanitized_name  # This changes the name before saving

            # Save image once here
            image_path = save_uploaded_image(image)
            print(f"Image saved to: {image_path}")  # Debugging

            try:
                # 1) Check for motorcycle
                mc_results = get_api_detections(image_path, settings.MOTORCYCLE_API_URL)
                # Only proceed if at least one detection has class_name == 'motorcycle'
                has_motorcycle = any(r['class_name'] == 'motorcycle' for r in mc_results)
                if not has_motorcycle:
                    error_message = "No motorycle violations can be detected."
                    return render(request, 'motorcycle_violation/report_violation.html', {
                        'form': form,
                        'error_message': error_message
                    })

                # 2) Check for violations (only if motorcycle found)
                vio_results = get_api_detections(image_path, settings.VIOLATION_API_URL)
                if not vio_results:
                    error_message = "No violations can be detected."
                    return render(request, 'motorcycle_violation/report_violation.html', {
                        'form': form,
                        'error_message': error_message
                    })

                img_annotated = cv2.imread(image_path)
                img_raw = cv2.imread(image_path)

                # Process detected violations, crop images, and detect license plates
                detected_violations, cropped_image_filenames, license_plate_crops = process_detected_violations(
                    vio_results, img_annotated, img_raw, image
                )

                # Extract text from upscaled license plate images
                license_plate_texts = license_plate_img_to_text(license_plate_crops)

                # Save annotated image
                annotated_image_path = save_annotated_image(img_annotated, image.name)

                # Prepare relative paths for the template
                uploaded_image_path = f'uploads/{image.name}'
                annotated_image_path_relative = f'uploads/annotated_{image.name}'

                # Zip detected violations, cropped image filenames, license plate crops, and license plate texts
                zipped_data = zip(detected_violations, cropped_image_filenames, license_plate_crops, license_plate_texts)

                # Render the results in the template
                return render(request, 'motorcycle_violation/report_violation.html', {
                    'form': ReportViolationForm(),
                    'image_path': uploaded_image_path,
                    'annotated_image_path': annotated_image_path_relative,
                    'zipped_data': zipped_data,
                })

            except Exception as e:
                error_message = "Violation detection failed: Please try again later."

    return render(request, 'motorcycle_violation/report_violation.html', {
        'form': ReportViolationForm(),
        'error_message': error_message
    })


@login_required
def process_license_plate(request):
    if request.method == 'POST':
        license_plate = request.POST.get('license_plate')
        detected_violation = request.POST.get('detected_violation')
        cropped_image_path = request.POST.get('cropped_image_path')  # This should be the relative path

        # Debugging output
        print(f"License Plate: {license_plate}")
        print(f"Detected Violation: {detected_violation}")
        print("Cropped image path:", cropped_image_path)  # Check the path being saved

        # Check for empty fields
        if not license_plate or not detected_violation or not cropped_image_path:
            return JsonResponse({'success': False, 'message': 'All fields are required.'})

        try:
            # Attempt to retrieve the motorcycle by license plate
            motorcycle = Motorcycle.objects.get(registration_number=license_plate)

            # Create a Violation entry
            violation = Violation(
                reporter=request.user.citizen_profile,
                motorcycle=motorcycle,
                violation_type=detected_violation,
                license_plate=license_plate,
                is_downloaded=False,
                cropped_image=cropped_image_path  # Save the relative path to the cropped image
            )
            violation.save()

            # Return a JSON response with success message
            return JsonResponse({'success': True, 'message': 'Violation report successfully submitted!'})
        except Motorcycle.DoesNotExist:
            print(f"Motorcycle with license plate {license_plate} does not exist.")
            return JsonResponse({'success': False, 'message': 'Motorcycle with this license plate does not exist.'})
        except Exception as e:
            print(f"Unexpected error: {e}")
            return JsonResponse({'success': False, 'message': 'An unexpected error occurred. Please try again.'})

    # Fallback for non-POST requests
    return JsonResponse({'success': False, 'message': 'Invalid request method.'})



@login_required
def report_history(request):
    if request.user.user_type != 'citizen':
        messages.error(request, "Only citizens can access the report history.")
        return redirect('landing_page')  

    try:
        user_profile = request.user.citizen_profile
    except Exception:
        messages.error(request, "Citizen profile not found.")
        return redirect('landing_page')

    violations = Violation.objects.filter(reporter=user_profile).order_by('-timestamp')

    return render(request, 'motorcycle_violation/report_history.html', {
        'violations': violations
    })


@login_required
def manage_motorbikes(request):
    user_motorcycles = request.user.citizen_profile.motorcycles.all()  # Get user's motorcycles
    form = MotorcycleForm()
    if request.method == 'POST':
        if 'add_motorbike' in request.POST:
            form = MotorcycleForm(request.POST)
            if form.is_valid():
                motorcycle = form.save(commit=False)
                motorcycle.owner = request.user.citizen_profile  # Set owner to the current user
                motorcycle.save()
                messages.success(request, 'Motorbike added successfully!')  # Add success message
                return redirect('manage_motorbikes')  # Redirect to the same page after adding
        elif 'remove_motorbike' in request.POST:
            motorcycle_id = request.POST.get('motorcycle_id')
            try:
                motorcycle = Motorcycle.objects.get(id=motorcycle_id, owner=request.user.citizen_profile)
                motorcycle.delete()
                messages.success(request, 'Motorbike removed successfully!')  # Add success message
                return redirect('manage_motorbikes')  # Redirect after removing
            except Motorcycle.DoesNotExist:
                messages.error(request, 'Motorbike not found.')  # Handle not found case

    else:
        form = MotorcycleForm()  # Create a blank form for adding motorcycles

    return render(request, 'motorcycle_violation/manage_motorbikes.html', {
        'form': form,
        'motorcycles': user_motorcycles,
    })


@login_required
def see_all_motorbikes(request):
    # Get the user's profile
    user_profile = request.user.citizen_profile
    
    # Fetch all motorcycles associated with the user's profile
    motorbikes = Motorcycle.objects.filter(owner=user_profile)  

    return render(request, 'motorcycle_violation/see_all_motorbikes.html', {
        'motorbikes': motorbikes
    })


@login_required
def appeal_history(request):
    # Get the citizen profile of the logged-in user
    citizen_profile = request.user.citizen_profile
    # Fetch all appeals related to the citizen, ordered by creation date (newest first)
    appeals = Appeal.objects.filter(citizen=citizen_profile).order_by('-created_at')
    
    # Pass the appeals to the template
    return render(request, 'motorcycle_violation/appeal_history.html', {'appeals': appeals})


@login_required
def appeals(request):
    return render(request, 'motorcycle_violation/appeals.html')

@login_required
def fine_history(request):
    citizen_profile = get_object_or_404(CitizenProfile, user=request.user)
    fines = Violation.objects.filter(motorcycle__owner=citizen_profile).order_by('-timestamp')

    return render(request, 'motorcycle_violation/fine_history.html', {'fines': fines})


@login_required
def download_receipt(request):
    user = request.user
    if hasattr(user, 'citizen_profile'):
        citizen_profile = user.citizen_profile

        # Get all motorcycles with their license plates and fine balances
        motorcycles = [
            {'motorcycle': motorcycle, 'fine_balance': motorcycle.fine_balance}
            for motorcycle in citizen_profile.motorcycles.all()
            if motorcycle.fine_balance > 0  # Only include motorcycles with fines
        ]

        # Calculate current fines and credits
        current_fines = sum(m['fine_balance'] for m in motorcycles)
        current_credits = citizen_profile.credit_balance

        # Calculate suspension fines (assumes a property exists on CitizenProfile)
        suspension_fines = citizen_profile.total_suspension_fines

        # Calculate payable amount including suspension fines
        payable_amount = (current_fines + suspension_fines) - current_credits

        # If no fines, credits, and no suspension fines, show a message instead of generating a receipt
        if current_fines == 0 and current_credits == 0 and suspension_fines == 0:
            context = {
                'no_receipt': True,
            }
            return render(request, 'motorcycle_violation/receipt.html', context)

        # Generate a unique receipt ID
        receipt_id = str(uuid.uuid4())

        # Create and save the receipt with suspension fines
        receipt = Receipt(
            user=citizen_profile,
            current_fines=current_fines,
            current_credits=current_credits,
            suspension_fines=suspension_fines,
            payable_amount=payable_amount,
            receipt_id=receipt_id
        )
        receipt.save()
    
        # Save motorcycle fines associated with the receipt
        for motorcycle in motorcycles:
            MotorcycleReceipt.objects.create(
                receipt=receipt,
                motorcycle=motorcycle['motorcycle'],
                fine_balance=motorcycle['fine_balance']
            )

        # Reset motorcycle fines and credits after generating the receipt
        for motorcycle in citizen_profile.motorcycles.all():
            motorcycle.fine_balance = 0
            motorcycle.save()

        citizen_profile.credit_balance = 0
        citizen_profile.save()

        # Mark related violations as downloaded
        Violation.objects.filter(motorcycle__owner=citizen_profile, is_downloaded=False).update(is_downloaded=True)

        # Mark suspensions as paid
        Suspension.objects.filter(citizen=citizen_profile, is_paid=False).update(is_paid=True)

        # Generate QR Code with a link to the verification page
        qr_data = f"{request.build_absolute_uri('/motorcycle_violation/verify_receipt/')}{receipt_id}/"
        qr = qrcode.make(qr_data)
        qr_io = BytesIO()
        qr.save(qr_io, format="PNG")
        qr_io.seek(0)

        # Encode QR for inline display in HTML templates
        qr_base64 = base64.b64encode(qr_io.getvalue()).decode("utf-8")
        qr_image_data = f"data:image/png;base64,{qr_base64}"

        # Build context for the receipt template
        context = {
            'no_receipt': False,
            'motorcycles': motorcycles,
            'current_fines': current_fines,
            'current_credits': current_credits,
            'suspension_fines': suspension_fines,
            'payable_amount': abs(payable_amount),
            'receipt_id': receipt_id,
            'created_at': receipt.created_at,
            'status': 'Payable' if (current_fines + suspension_fines) > current_credits else 'Redeemable',
            'user_name': citizen_profile.user.username,
            'user_email': citizen_profile.user.email,
            'user_cnic': citizen_profile.cnic,
            'qr_image_data': qr_image_data,
            'receipt_status': receipt.status,
            'status_updated_at': receipt.status_updated_at if receipt.status_updated_at else "N/A",
        }

        return render(request, 'motorcycle_violation/receipt.html', context)

    return JsonResponse({'error': 'User does not have a citizen profile.'}, status=400)



@login_required
def verify_receipt(request, receipt_id):
    try:
        receipt = Receipt.objects.get(receipt_id=receipt_id)
        
        # Fetch the associated motorcycle receipts
        motorcycle_receipts = receipt.motorcycle_receipts.all()

        context = {
            'receipt': receipt,
            'motorcycle_receipts': motorcycle_receipts,
            'verified': True,
            'receipt_status': receipt.status,
            'status_updated_at': receipt.status_updated_at if receipt.status_updated_at else "N/A",
            'suspension_fines': receipt.suspension_fines,
        }

        return render(request, 'motorcycle_violation/verify_receipt.html', context)

    except Receipt.DoesNotExist:
        return render(request, 'motorcycle_violation/verify_receipt.html', {'verified': False})




@login_required
def download_receipt_pdf(request, receipt_id):
    # Fetch the specific receipt using the receipt_id
    receipt = get_object_or_404(Receipt, receipt_id=receipt_id)

    # Generate QR Code
    qr_data = f"{request.build_absolute_uri('/motorcycle_violation/verify_receipt/')}{receipt_id}/"
    qr = qrcode.make(qr_data)
    qr_io = BytesIO()
    qr.save(qr_io, format="PNG")
    qr_io.seek(0)

    # Encode QR for inline display in HTML templates
    qr_base64 = base64.b64encode(qr_io.getvalue()).decode("utf-8")
    qr_image_data = f"data:image/png;base64,{qr_base64}"

    # Get the user profile
    user_profile = receipt.user

    # Retrieve motorcycles and their fines associated with the receipt
    motorcycles = [
        {
            'license_plate': motorcycle_receipt.motorcycle.registration_number,
            'fine_balance': motorcycle_receipt.fine_balance,
            'model': motorcycle_receipt.motorcycle.model_name
        }
        for motorcycle_receipt in receipt.motorcycle_receipts.all() if motorcycle_receipt.fine_balance > 0
    ]

    # Prepare context with receipt details, user information, and motorcycle details
    context = {
        'current_fines': receipt.current_fines,
        'current_credits': receipt.current_credits,
        'suspension_fines': receipt.suspension_fines,
        'payable_amount': abs(receipt.payable_amount),
        'receipt_id': receipt.receipt_id,
        'created_at': receipt.created_at,
        'status': 'Payable' if (receipt.current_fines + receipt.suspension_fines) > receipt.current_credits else 'Redeemable',
        'user_name': user_profile.user.username,
        'user_email': user_profile.user.email,
        'user_cnic': user_profile.cnic,
        'motorcycles': motorcycles,
        'qr_image_data': qr_image_data,
        'receipt_status': receipt.status,
        'status_updated_at': receipt.status_updated_at if receipt.status_updated_at else "N/A",
    }

    # Render the HTML to string for PDF generation
    html_string = render_to_string('motorcycle_violation/receipt_pdf.html', context)

    # Create the PDF response
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="receipt_{receipt_id}.pdf"'

    # Generate PDF using WeasyPrint with base_url for static file resolution
    HTML(string=html_string, base_url=request.build_absolute_uri('/')).write_pdf(response)

    return response



@login_required
def receipt_history(request):
    user_profile = request.user.citizen_profile  # Assuming user is a citizen
    receipts = Receipt.objects.filter(user=user_profile).order_by('-created_at')  # Get user's receipts and order by creation date

    return render(request, 'motorcycle_violation/receipt_history.html', {'receipts': receipts})

@login_required  # Ensure the user is logged in
def file_appeal(request):
    citizen_profile = request.user.citizen_profile  # Get the CitizenProfile of the logged-in user

    # Get violations for the user that are invalid and have not been appealed, sorted by timestamp (newest first)
    violations = Violation.objects.filter(motorcycle__owner=citizen_profile, is_downloaded=False, is_appealed='no').order_by('-timestamp')

    if request.method == 'POST':
        selected_violation_id = request.POST['violation']  # Get the selected violation ID
        reason = request.POST['reason']

        try:
            # Retrieve the violation to update it
            violation = Violation.objects.get(id=selected_violation_id)

            # Create an Appeal instance and save it to the database
            appeal = Appeal(
                citizen=citizen_profile,
                violation=violation,
                reason=reason,
                status='pending',  # Initial status of the appeal
            )
            appeal.save()

            # Update the violation's is_appealed status to 'yes'
            violation.is_appealed = 'yes'
            violation.save()

            messages.success(request, 'Your appeal has been submitted successfully.')  # Success message
            return redirect('file_appeal')  # Redirect to a success page or back to the dashboard

        except Violation.DoesNotExist:
            messages.error(request, 'The selected violation does not exist.')  # Error message for invalid violation
            return redirect('file_appeal')  # Redirect back to the appeal page

    return render(request, 'motorcycle_violation/file_appeal.html', {'violations': violations})  # Pass violations to the template



@login_required
def review_appeals(request):
    appeals = Appeal.objects.filter(status='pending').order_by('created_at')

    if request.method == 'POST':
        appeal_id = request.POST.get('appeal_id')
        action = request.POST.get('action')  # Expected: 'accept' or 'reject'
        
        try:
            appeal = Appeal.objects.get(id=appeal_id)
            violation = appeal.violation
            motorcycle = violation.motorcycle
            # The appealing citizen owns the motorcycle (used for credit/fine adjustment)
            appealing_citizen = appeal.citizen
            # Suspension will be applied to the reporter
            reporter_profile = violation.reporter

            if action == 'accept':
                suspend_option = request.POST.get('suspend_option', 'no')
                suspension_days = request.POST.get('suspension_days')
                if suspend_option == 'yes' and suspension_days:
                    try:
                        suspension_days = float(suspension_days)
                    except ValueError:
                        suspension_days = 3
                else:
                    suspension_days = None

                suspension_reason = request.POST.get(
                    'suspension_reason', 'Appeal accepted with suspension.'
                )

                # Adjust credit balance and fine balance
                if violation.is_downloaded:
                    appealing_citizen.credit_balance += violation.fine_amount
                    appealing_citizen.save()
                else:
                    motorcycle.fine_balance -= violation.fine_amount
                    motorcycle.save()

                appeal.status = 'accepted'
                violation.is_appealed = 'accepted'
                appeal.officer = request.user.officer_profile

                if suspension_days is not None:
                    # Get the suspension fine from the select field.
                    suspension_fine = request.POST.get('suspension_fine')
                    try:
                        suspension_fine = float(suspension_fine)
                    except ValueError:
                        suspension_fine = 100  # default if invalid

                    # Create a suspension for the reporter
                    suspension = Suspension(
                        citizen=reporter_profile,
                        violation=violation,
                        reason=suspension_reason,
                        suspended_by=request.user.officer_profile,
                        fine=suspension_fine
                    )
                    suspension.save(provided_duration=suspension_days)
                    messages.success(request, 'Appeal accepted and reporter suspended successfully.')
                else:
                    messages.success(request, 'Appeal accepted successfully.')
            else:
                appeal.status = 'rejected'
                violation.is_appealed = 'rejected'
                appeal.officer = request.user.officer_profile
                messages.info(request, 'Appeal rejected successfully.')

            appeal.save()
            violation.save()

            return redirect('review_appeals')

        except Appeal.DoesNotExist:
            messages.error(request, 'Appeal not found.')
    
    return render(request, 'motorcycle_violation/review_appeals.html', {'appeals': appeals})




@login_required
def reviewed_appeal_history(request):
    # Fetch all reviewed appeals related to the logged-in officer
    reviewed_appeals = Appeal.objects.filter(officer=request.user.officer_profile, status__in=['accepted', 'rejected']).order_by('-updated_at')
    
    return render(request, 'motorcycle_violation/reviewed_appeal_history.html', {'reviewed_appeals': reviewed_appeals})



@login_required
def upload_video(request):
    """
    Allows a logged-in user to upload a video for motorcycle violation detection.

    - Converts video to 15 FPS using OpenCV.
    - Runs `main_pipeline()` to detect traffic violations.
    - Displays violation frames, license plate crops, and details in the template.
    - Suspended users are redirected with an error.

    Returns:
        Rendered HTML page with success/error messages, processed video, and violation data.
    """
    # Check if the citizen is currently suspended
    if request.user.citizen_profile.is_currently_suspended:
        messages.error(request, "You are currently suspended and cannot report a violation.")
        return redirect('citizen_dashboard')
    
    if request.method == 'POST':
        video_file = request.FILES.get('video_file')
        if not video_file:
            return render(request, 'motorcycle_violation/upload_video.html', {
                'error': 'No video file uploaded.'
            })

        # Save the uploaded video after converting it to 15fps.
        video_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
        os.makedirs(video_dir, exist_ok=True)
        input_video_path = os.path.join(video_dir, video_file.name)

        # Write the uploaded video to a temporary file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as temp_file:
            for chunk in video_file.chunks():
                temp_file.write(chunk)
            temp_video_path = temp_file.name

        # Open the temporary video file with OpenCV.
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return render(request, 'motorcycle_violation/upload_video.html', {
                'error': 'Failed to process the uploaded video.'
            })

        # Get original fps to compute sampling interval
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            orig_fps = 30.0  # or any sane default
        frame_interval = max(1, int(round(orig_fps / 15.0)))

        # Get frame size for the writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define codec and create VideoWriter with 15 fps metadata
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(input_video_path, fourcc, 15.0, (width, height))

        # Read frames and write only every frame_interval-th frame
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        os.remove(temp_video_path)

        # Run the processing pipeline using the uploaded video.
        start_time = time.time()
        result = main_pipeline(input_video_path)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{frame_idx} frames completed in {elapsed:.2f} seconds.")

        # Handle no-violation message
        if isinstance(result, str) and result == "No violations can be detected from this video.":
            # Compute the URL for the processed (converted) video.
            output_video_filename = f"{os.path.splitext(video_file.name)[0]}_out_converted.mp4"
            output_video_url = os.path.join(
                settings.MEDIA_URL.rstrip('/'),
                'output_videos',
                output_video_filename
            ).replace('\\', '/')
            return render(request, 'motorcycle_violation/upload_video.html', {
                'violation_data': [],
                'output_video': output_video_url,
                'success': result
            })

        # Expecting the pipeline to return a tuple: (violation_dir, violation_df)
        pipeline_result, violation_df = result

        # Check if pipeline returned an error message.
        # Only treat it as a message if it's not a valid directory.
        if isinstance(pipeline_result, str) and not os.path.isdir(pipeline_result):
            return render(request, 'motorcycle_violation/upload_video.html', {
                'message': pipeline_result
            })

        # At this point, pipeline_result is assumed to be the directory with violation images.
        violation_dir = pipeline_result

        # Compute the URL for the processed (converted) video.
        # Using the correct converted file name (_out_converted.mp4).
        output_video_filename = f"{os.path.splitext(video_file.name)[0]}_out_converted.mp4"
        output_video_url = os.path.join(
            settings.MEDIA_URL.rstrip('/'),
            'output_videos',
            output_video_filename
        ).replace('\\', '/')

        # Build a list of violation data from violation_df.
        # Images were saved with the following naming conventions:
        # - Violation image: "track_{track_id}_frame_{frame}.jpg"
        # - License plate (LP) crop image: "track_{track_id}_frame_{frame}_lp.jpg"
        violation_data = []
        for index, row in violation_df.iterrows():
            track_id = row["track_id"]
            frame_number = row["frame"]

            # Construct the expected file names.
            violation_filename = f"track_{track_id}_frame_{frame_number}.jpg"
            lp_crop_filename = f"track_{track_id}_frame_{frame_number}_lp.jpg"

            # Compute the relative directory for the violation images.
            rel_violation_dir = os.path.relpath(violation_dir, settings.MEDIA_ROOT)

            # Build the URLs for both the violation image and the LP crop image.
            image_url = os.path.join(
                settings.MEDIA_URL.rstrip('/'),
                rel_violation_dir,
                violation_filename
            ).replace('\\', '/')
            lp_crop_url = os.path.join(
                settings.MEDIA_URL.rstrip('/'),
                rel_violation_dir,
                lp_crop_filename
            ).replace('\\', '/')

            violation_class = row["violation_class"]
            violation_conf = row["violation_conf"]
            best_lp_text = row["best_lp_text"]

            violation_data.append({
                "image_url": image_url,
                "lp_crop_url": lp_crop_url,
                "violation_class": violation_class,
                "violation_conf": f"{violation_conf:.2f}",
                "best_lp_text": best_lp_text
            })

        # Prepare and render the template.
        context = {
            'violation_data': violation_data,
            'output_video': output_video_url,
            'success': "Violations detected!" if violation_data else "No violations detected."
        }
        return render(request, 'motorcycle_violation/upload_video.html', context)
    
    return render(request, 'motorcycle_violation/upload_video.html')




@login_required
def report_license_plate(request):
    if request.method == "POST":
        # Retrieve the license plate and image URL from POST data.
        license_plate = request.POST.get('license_plate', '').strip()
        image_url = request.POST.get('image_url', '').strip()
        if image_url.startswith(settings.MEDIA_URL):
            image_url = image_url[len(settings.MEDIA_URL):]
        # Use a default detected violation value (adjust as needed).
        detected_violation = request.POST.get('detected_violation', 'Unknown Violation')
        
        # Validate that a license plate was provided.
        if not license_plate:
            return JsonResponse({'success': False, 'message': 'License plate is required.'})
        
        try:
            # Attempt to retrieve the motorcycle by license plate.
            motorcycle = Motorcycle.objects.get(registration_number=license_plate)
            
            # Create a Violation entry.
            violation = Violation(
                reporter=request.user.citizen_profile,
                motorcycle=motorcycle,
                violation_type=detected_violation,
                license_plate=license_plate,
                is_downloaded=False,
                cropped_image=image_url  # Save the relative path to the cropped image.
            )
            violation.save()

            # Optionally, you can update fine balances and credits here.

            # Return a JSON response with a success message.
            return JsonResponse({'success': True, 'message': 'Violation report successfully submitted!'})
        except Motorcycle.DoesNotExist:
            # Motorcycle not found.
            return JsonResponse({'success': False, 'message': 'Motorcycle with this license plate does not exist.'})
        except Exception as e:
            # Log or print the error as needed.
            print(f"Unexpected error: {e}")
            return JsonResponse({'success': False, 'message': 'An unexpected error occurred. Please try again.'})
    
    # For non-POST requests, return an error.
    return JsonResponse({'success': False, 'message': 'Invalid request method.'})


@login_required
def clear_media_redirect(request):
    try:
        # Remove and recreate the "output_videos" folder.
        output_videos_path = os.path.join(settings.MEDIA_ROOT, 'output_videos')
        if os.path.exists(output_videos_path):
            shutil.rmtree(output_videos_path)
        os.makedirs(output_videos_path)

        # Remove and recreate the "videos" folder.
        videos_path = os.path.join(settings.MEDIA_ROOT, 'videos')
        if os.path.exists(videos_path):
            shutil.rmtree(videos_path)
        os.makedirs(videos_path)

        # Remove and recreate the "videos" folder.
        csv_path = os.path.join(settings.MEDIA_ROOT, 'csv')
        if os.path.exists(csv_path):
            shutil.rmtree(csv_path)
        os.makedirs(csv_path)
    
        print("Media folders cleared successfully.")
    except Exception as e:
        print(f"Error clearing media: {e}")
    # Redirect back to the upload page.
    return redirect('upload_video')  # adjust the URL name as needed


def receipt_list(request):
    """
    Searches for a user by their exact username.
    If a matching user is found, fetch that user’s receipts with status 'GENERATED'.
    Optionally, if a 'receipt_q' is provided, filter those receipts by receipt_id.
    
    When a receipt search is performed (receipt_q provided) and no receipt is found,
    an error message is displayed and the “User found” message is not shown.
    """
    user_query = request.GET.get('user_query', '').strip()
    receipt_query = request.GET.get('receipt_q', '').strip()
    user_obj = None
    receipts = None

    if user_query:
        # Look for a CitizenProfile whose related user's username exactly matches the query.
        user_obj = CitizenProfile.objects.select_related('user').filter(user__username=user_query).first()
        if not user_obj:
            messages.error(request, f"No user found with username '{user_query}'.")
        else:
            # Get receipts with status 'GENERATED' for this user.
            receipts = Receipt.objects.filter(user=user_obj, status='GENERATED')
            if receipt_query:
                receipts = receipts.filter(receipt_id__icontains=receipt_query)
                if not receipts.exists():
                    messages.error(request, f"No receipt found with ID containing '{receipt_query}'.")
            else:
                messages.success(request, f"User '{user_query}' found.")

    context = {
        'user_query': user_query,
        'receipt_query': receipt_query,
        'user_obj': user_obj,
        'receipts': receipts,
    }
    return render(request, 'motorcycle_violation/receipt_list.html', context)


def update_receipt_status(request, receipt_id):
    """
    Updates the receipt status:
      - If payable_amount is negative: update status to 'REDEEMED'
      - Otherwise: update status to 'PAID'
    Only receipts with current status 'GENERATED' will be updated.

    If the request is made via AJAX, returns a JSON response with the new button text
    and CSS class so that the client can update the page without a reload.
    """
    receipt = get_object_or_404(Receipt, id=receipt_id)
    
    if request.method == "POST" and receipt.status == 'GENERATED':
        if receipt.payable_amount < 0:
            receipt.status = 'REDEEMED'
            button_text = "Redeemed"
            # Change the button appearance after redemption.
            button_class = "btn btn-secondary btn-sm"
        else:
            receipt.status = 'PAID'
            button_text = "Paid"
            # Change the button appearance after payment.
            button_class = "btn btn-secondary btn-sm"
        receipt.save()
        
        # If this is an AJAX request, return JSON.
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'success': True,
                'new_status': receipt.status,
                'button_text': button_text,
                'button_class': button_class,
                'receipt_id': receipt.receipt_id,
            })
        messages.success(request, f"Receipt {receipt.receipt_id} updated successfully to {receipt.status.capitalize()}.")
    else:
        msg = "Receipt is not in a valid state to update."
        messages.warning(request, msg)
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'success': False, 'message': msg})
    
    return redirect('receipt_list')


def suspension_overview(request):
    # Assuming the logged-in user has a related CitizenProfile via "citizen_profile"
    citizen = request.user.citizen_profile
    now = timezone.now()
    active_suspensions = citizen.suspension_history.filter(suspended_until__gt=now)
    context = {}

    if active_suspensions.exists():
        # Get the currently active suspension (the one with the latest suspended_until)
        active_suspension = active_suspensions.order_by('-suspended_until').first()
        remaining = active_suspension.suspended_until - now
        remaining_seconds = int(remaining.total_seconds())
        total_fine = citizen.total_suspension_fines

        context.update({
            'active_suspension': active_suspension,
            'remaining_seconds': remaining_seconds,
            'total_fine': total_fine,
        })

    return render(request, 'motorcycle_violation/suspension_overview.html', context)




@login_required
def suspension_history(request):
    """
    Display the suspension history (active and past) for the logged‐in citizen.
    Only citizens can access this view.
    """
    if not hasattr(request.user, 'citizen_profile'):
        raise PermissionDenied("Only citizens can view suspension history.")
        
    suspensions = request.user.citizen_profile.suspension_history.all().order_by('-suspended_at')
    return render(request, 'motorcycle_violation/suspension_history.html', {
        'suspensions': suspensions
    })


@login_required
@csrf_exempt  # Disable CSRF protection
def chatbot_api(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_message = data.get("message", "").strip().lower()

        # Greeting if the user says hi/hello
        if user_message in ["hi", "hello", "hey"]:
            reply = "Hi, how are you doing? Type 'help' for a list of keywords."
        
        # Show keywords when user types 'help'
        elif user_message == "help":
            reply = (
                "Here are some keywords you can ask about:<br>"
                "1. 'hours' - Our support hours.<br>"
                "2. 'contact' - How to contact us.<br>"
                "3. 'credits' - Information about credits.<br>"
                "4. 'receipt' - How to download your receipt.<br>"
                "5. 'office' - Local Safe Ride office details.<br>"
                "6. 'suspensions' - Why suspensions occur.<br>"
                "7. 'license plate' - License plate format details.<br>"
                "8. 'avoid suspensions' - Tips on avoiding suspensions.<br>"
                "9. 'appeal' - How to appeal if you're wrongly fined.<br>"
                "10. 'vision tech' - Info about our computer vision technology.<br>"
                "11. 'violations' - Details about violation types, fines, credits, and suspension policies.<br>"
                "Try asking one of these keywords!"
            )
        
        elif "hours" in user_message:
            reply = "Our support hours are 9am to 5pm, Monday to Friday."
        
        elif "contact" in user_message:
            reply = "You can contact us at info@saferide.com."
        
        elif "credits" in user_message:
            reply = (
                "Credits can be redeemed at your local Safe Ride office, while fines must be paid. "
                "For every successful report, you earn 100 credits."
            )
        
        elif "receipt" in user_message:
            reply = "You can download your receipt from the Receipt Management section on your dashboard."
        
        elif "office" in user_message:
            reply = "Please visit your local Safe Ride office for credit redemption or further assistance."
        
        elif "suspensions" in user_message:
            reply = (
                "Suspensions occur when duplicate, inaccurate, or false violation reports are submitted. "
                "The suspension amount and duration may change based on your past history of actions. "
                "Suspension amounts and durations change based on your past history of actions. "
                "Always ensure your report is accurate to avoid unnecessary suspensions."
            )
        
        elif "license plate" in user_message:
            reply = (
                "The license plate format should be: ABC-1234-17-A<br>"
                "• 3 mandatory letters<br>"
                "• 3 or 4 mandatory digits<br>"
                "• 2 optional digits<br>"
                "• 1 optional letter"
            )
        
        elif "avoid suspensions" in user_message or "how to avoid suspensions" in user_message:
            reply = (
                "How to Avoid Suspensions:<br>"
                "• Do not report a violation if the license plate on the vehicle is different from the one on record.<br>"
                "• Do not report a violation if the license plate is not clearly visible.<br>"
                "• Do not report a violation if there is no actual violation observed.<br>"
                "• Do not report the same violation multiple times.<br>"
                "Always ensure your reports are accurate and supported by clear evidence."
            )
        
        elif "appeal" in user_message:
            reply = (
                "If you have been wrongly fully fined, you can file an appeal. Your appeal will be reviewed by our officers, "
                "and if approved, the fine may be adjusted or waived."
            )
        
        elif "vision tech" in user_message or "computer vision" in user_message:
            reply = (
                "Our system uses state-of-the-art computer vision technologies to detect violations. "
                "This advanced system helps ensure accurate and fair reporting."
            )
        
        elif "violations" in user_message:
            reply = (
                "Here’s what you need to know about violations:<br>"
                "• Violations include: wearing no helmet (for driver or passenger) and carrying more than 2 riders.<br>"
                "• Fine amount: 500 PKR for each violation.<br>"
                "• For every successful report, you earn 100 credits.<br>"
            )
        
        # Default fallback message
        else:
            reply = "Sorry, I don't understand that. Please type 'help' for a list of keywords or rephrase your question."
        
        return JsonResponse({"reply": reply})
    return JsonResponse({"reply": "Invalid request."}, status=400)
