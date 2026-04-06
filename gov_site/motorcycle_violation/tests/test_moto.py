from django.contrib.auth import get_user_model  # This is the correct way to get the custom user model
from django.urls import reverse
from django.test import TestCase
from motorcycle_violation.models import Motorcycle, CitizenProfile

class ManageMotorbikesTests(TestCase):

    def setUp(self):
        # Use the custom user model to create a user
        User = get_user_model()
        self.user = User.objects.create_user(username='testuser', password='password')
        self.user_profile = CitizenProfile.objects.create(user=self.user)
        self.url = reverse('manage_motorbikes')  # URL for the 'manage_motorbikes' view

        # Log in the user
        self.client.login(username='testuser', password='password')

    def test_add_motorbike_success(self):
        data = {
            'registration_number': 'ABC-1234-12-A',
            'model_name': 'Yamaha R1',
            'fine_balance': 0.00,
            'add_motorbike': 'Add Motorbike',  # This triggers the form submission for adding
        }

        response = self.client.post(self.url, data)

        # Check if the motorcycle is added to the database
        self.assertTrue(
            Motorcycle.objects.filter(registration_number='ABC-1234-12-A').exists(),
            "Motorcycle was not added."
        )

        # Ensure we're redirected back to the manage motorbikes page
        self.assertRedirects(response, self.url)

    def test_remove_motorbike_success(self):
        # Create a motorcycle associated with the user
        motorcycle = Motorcycle.objects.create(
            registration_number="ABC-1234-12-A",
            model_name="Yamaha R1",
            fine_balance=0.00,
            owner=self.user_profile  # Assign the user profile as the owner
        )

        data = {
            'motorcycle_id': motorcycle.id,  # Pass the motorcycle ID for removal
            'remove_motorbike': 'Remove Motorbike',  # This triggers the form submission for removal
        }

        response = self.client.post(self.url, data)

        # Check if the motorcycle is removed from the database
        self.assertFalse(
            Motorcycle.objects.filter(id=motorcycle.id).exists(),
            "Motorcycle was not removed."
        )

        # Ensure we're redirected back to the manage motorbikes page
        self.assertRedirects(response, self.url)
