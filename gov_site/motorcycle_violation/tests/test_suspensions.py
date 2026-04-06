# motorcycle_violation/tests/test_suspensions.py

import datetime
from unittest.mock import patch
from decimal import Decimal

from django.test import TestCase, Client
from django.urls import reverse
from django.utils import timezone

from motorcycle_violation.models import (
    CustomUser, Motorcycle, Violation, Suspension
)


class SuspensionOverviewTests(TestCase):
    def setUp(self):
        # Create a citizen and log in
        self.client = Client()
        self.citizen = CustomUser.objects.create_user(
            username='citizen', password='pass',
            user_type='citizen', email='c@example.com'
        )
        self.profile = self.citizen.citizen_profile
        self.client.login(username='citizen', password='pass')

        # Need one Violation (for linking Suspension)
        self.mc = Motorcycle.objects.create(
            owner=self.profile,
            registration_number='ABC-100', model_name='TestBike'
        )
        self.violation = Violation.objects.create(
            reporter=self.profile,
            motorcycle=self.mc,
            violation_type='test_violation',
            license_plate=self.mc.registration_number,
            cropped_image='dummy.jpg'
        )

        self.url = reverse('suspension_overview')

        # A fixed "now" for deterministic tests
        self.fixed_now = timezone.make_aware(
            datetime.datetime(2025, 4, 20, 12, 0, 0),
            timezone.get_current_timezone()
        )

    def test_no_active_suspensions(self):
        """When there are no future suspensions, context stays empty."""
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn('active_suspension', resp.context)
        self.assertNotIn('remaining_seconds', resp.context)
        self.assertNotIn('total_fine', resp.context)

    from decimal import Decimal

    @patch('django.utils.timezone.now')
    def test_active_suspension_shown(self, mock_now):
        """
        When there is a suspension whose `suspended_until` > now,
        the view should include it plus remaining_seconds and total_fine.
        """
        mock_now.return_value = self.fixed_now

        # Create exactly one active suspension via bulk_create to bypass save()
        susp = Suspension(
            citizen=self.profile,
            violation=self.violation,
            reason='TestReason',
            suspended_by=None,
            suspended_at=self.fixed_now - datetime.timedelta(days=1),
            suspended_until=self.fixed_now + datetime.timedelta(hours=2),
            fine=Decimal('45.67'),  # Use Decimal here
            is_paid=False
        )
        Suspension.objects.bulk_create([susp])

        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

        # It should have injected these keys
        self.assertIn('active_suspension', resp.context)
        self.assertIn('remaining_seconds', resp.context)
        self.assertIn('total_fine', resp.context)

        # Check values
        self.assertEqual(resp.context['active_suspension'].reason, 'TestReason')
        expected_secs = int((datetime.timedelta(hours=2)).total_seconds())
        self.assertEqual(resp.context['remaining_seconds'], expected_secs)

        # Use Decimal for the fine comparison
        self.assertEqual(resp.context['total_fine'], Decimal('45.67'))



class SuspensionHistoryTests(TestCase):
    def setUp(self):
        # Citizen + two violations
        self.client = Client()
        self.citizen = CustomUser.objects.create_user(
            username='citizenH', password='pass',
            user_type='citizen', email='h@example.com'
        )
        self.profile = self.citizen.citizen_profile

        # First violation
        mc1 = Motorcycle.objects.create(
            owner=self.profile,
            registration_number='ABC-200', model_name='Bike1'
        )
        viol1 = Violation.objects.create(
            reporter=self.profile,
            motorcycle=mc1,
            violation_type='v1',
            license_plate=mc1.registration_number,
            cropped_image='dummy.jpg'
        )

        # Second violation
        mc2 = Motorcycle.objects.create(
            owner=self.profile,
            registration_number='ABC-201', model_name='Bike2'
        )
        viol2 = Violation.objects.create(
            reporter=self.profile,
            motorcycle=mc2,
            violation_type='v2',
            license_plate=mc2.registration_number,
            cropped_image='dummy.jpg'
        )

        # Two suspensions with different suspended_at times
        dt1 = timezone.make_aware(
            datetime.datetime(2025, 4, 18, 9, 0, 0),
            timezone.get_current_timezone()
        )
        dt2 = timezone.make_aware(
            datetime.datetime(2025, 4, 19, 10, 0, 0),
            timezone.get_current_timezone()
        )

        susp1 = Suspension(
            citizen=self.profile,
            violation=viol1,
            reason='reason1',
            suspended_by=None,
            suspended_at=dt1,
            suspended_until=dt1 + datetime.timedelta(days=1),
            fine=10,
            is_paid=False
        )
        susp2 = Suspension(
            citizen=self.profile,
            violation=viol2,
            reason='reason2',
            suspended_by=None,
            suspended_at=dt2,
            suspended_until=dt2 + datetime.timedelta(days=1),
            fine=20,
            is_paid=True
        )
        Suspension.objects.bulk_create([susp1, susp2])

        self.url = reverse('suspension_history')

    def test_history_for_citizen(self):
        """A logged‑in citizen sees all their suspensions ordered newest first."""
        self.client.login(username='citizenH', password='pass')
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

        suspensions = resp.context['suspensions']
        # Check ordering by suspended_at descending: reason2 then reason1
        reasons = [s.reason for s in suspensions]
        self.assertEqual(reasons, ['reason2', 'reason1'])

    def test_history_forbidden_for_officer(self):
        """Users without a citizen_profile (e.g. officers) get 403."""
        officer = CustomUser.objects.create_user(
            username='officerH', password='pass',
            user_type='officer', email='o@example.com'
        )
        self.client.login(username='officerH', password='pass')
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 403)
