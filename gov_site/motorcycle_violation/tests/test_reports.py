from decimal import Decimal
from django.test import TestCase, Client
from django.urls import reverse

from motorcycle_violation.models import CustomUser, Motorcycle, Violation


class ProcessLicensePlateTests(TestCase):
    def setUp(self):
        self.client = Client()
        # Create a citizen user and auto‑created profile
        self.user = CustomUser.objects.create_user(
            username='user1',
            password='pass',
            user_type='citizen',
            email='u1@example.com'
        )
        self.profile = self.user.citizen_profile
        # Log in
        self.client.force_login(self.user)

        # Create a motorcycle owned by this citizen
        self.mc = Motorcycle.objects.create(
            owner=self.profile,
            registration_number='PLT-1234',
            model_name='TestCycle'
        )

        # Record initial balances
        self.initial_credit = self.profile.credit_balance
        self.initial_fine = self.mc.fine_balance

        # Reverse lookup for the view
        self.url = reverse('process_license_plate')

    def test_get_invalid_method(self):
        """GET should return a JSON telling 'Invalid request method.'"""
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data['success'])
        self.assertEqual(data['message'], 'Invalid request method.')

    def test_post_missing_fields(self):
        """POST without all required fields should error out."""
        resp = self.client.post(self.url, {})  # no data
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data['success'])
        self.assertEqual(data['message'], 'All fields are required.')

    def test_post_nonexistent_motorcycle(self):
        """Using a plate that doesn't exist should return the motorcycle-not-found message."""
        resp = self.client.post(self.url, {
            'license_plate': 'NOT-FOUND',
            'detected_violation': 'speeding',
            'cropped_image_path': 'some/path.jpg'
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data['success'])
        self.assertEqual(
            data['message'],
            'Motorcycle with this license plate does not exist.'
        )

    def test_post_valid_creates_violation_and_updates_balances(self):
        """A valid POST should create a Violation and update balances."""
        resp = self.client.post(self.url, {
            'license_plate': self.mc.registration_number,
            'detected_violation': 'speeding',
            'cropped_image_path': 'uploads/crop1.jpg'
        })
        # 1) Check JSON response
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data['success'])
        self.assertEqual(
            data['message'],
            'Violation report successfully submitted!'
        )

        # 2) A Violation object was created
        v = Violation.objects.get(license_plate=self.mc.registration_number, violation_type='speeding')
        self.assertIsNotNone(v)
        self.assertEqual(v.cropped_image, 'uploads/crop1.jpg')
        self.assertFalse(v.is_downloaded)

        # 3) Reporter credit_balance increased by 100
        self.profile.refresh_from_db()
        self.assertEqual(
            self.profile.credit_balance,
            self.initial_credit + Decimal('100.00')
        )

        # 4) Motorcycle fine_balance increased by 500
        self.mc.refresh_from_db()
        self.assertEqual(
            self.mc.fine_balance,
            self.initial_fine + Decimal('500.00')
        )
