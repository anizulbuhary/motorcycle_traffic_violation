# motorcycle_violation/tests/test_appeals.py
import datetime
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.messages import get_messages
from django.utils import timezone

from motorcycle_violation.models import (
    CustomUser, CitizenProfile, OfficerProfile,
    Motorcycle, Violation, Appeal, Suspension
)


class AppealViewTests(TestCase):
    def setUp(self):
        # Citizen + profile
        self.citizen = CustomUser.objects.create_user(
            username='citizen1', password='pass',
            user_type='citizen', email='c1@example.com'
        )
        # Create motorcycle + violation (with dummy cropped_image name)
        self.mc = Motorcycle.objects.create(
            owner=self.citizen.citizen_profile,
            registration_number='ABC-1234', model_name='TestBike'
        )
        self.violation = Violation.objects.create(
            reporter=self.citizen.citizen_profile,
            motorcycle=self.mc,
            violation_type='no_helmet',
            license_plate=self.mc.registration_number,
            cropped_image='dummy.jpg'
        )

        self.client = Client()

    def test_file_appeal_get_and_post(self):
        self.client.login(username='citizen1', password='pass')
        url = reverse('file_appeal')

        # GET → should include our violation
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(self.violation, resp.context['violations'])

        # POST valid appeal
        resp = self.client.post(url, {
            'violation': str(self.violation.id),
            'reason': 'I had a permit'
        })
        self.assertEqual(resp.status_code, 302)

        # Verify Appeal created and Violation updated
        appeal = Appeal.objects.get(violation=self.violation)
        self.violation.refresh_from_db()
        self.assertEqual(appeal.reason, 'I had a permit')
        self.assertEqual(appeal.status, 'pending')
        self.assertEqual(self.violation.is_appealed, 'yes')

        # Success message
        msgs = [m.message for m in get_messages(resp.wsgi_request)]
        self.assertIn('Your appeal has been submitted successfully.', msgs)

        # POST invalid
        resp2 = self.client.post(url, {
            'violation': '9999', 'reason': 'none'
        })
        self.assertEqual(resp2.status_code, 302)
        msgs2 = [m.message for m in get_messages(resp2.wsgi_request)]
        self.assertIn('The selected violation does not exist.', msgs2)


class ReviewAppealsViewTests(TestCase):
    def setUp(self):
        # Citizen + violation + appeal
        self.citizen = CustomUser.objects.create_user(
            username='citizen2', password='pass',
            user_type='citizen', email='c2@example.com'
        )
        mc = Motorcycle.objects.create(
            owner=self.citizen.citizen_profile,
            registration_number='XYZ-5678', model_name='Bike2'
        )
        self.violation = Violation.objects.create(
            reporter=self.citizen.citizen_profile,
            motorcycle=mc,
            violation_type='speeding',
            license_plate=mc.registration_number,
            cropped_image='dummy.jpg'
        )
        self.appeal = Appeal.objects.create(
            citizen=self.citizen.citizen_profile,
            violation=self.violation,
            reason='Was in emergency'
        )

        # Officer
        self.officer = CustomUser.objects.create_user(
            username='officer2', password='pass',
            user_type='officer', email='o2@example.com'
        )
        self.client = Client()
        self.client.login(username='officer2', password='pass')

    def test_review_appeals_accept_and_reject(self):
        url = reverse('review_appeals')

        # GET pending
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(self.appeal, resp.context['appeals'])

        # Accept without suspension
        resp1 = self.client.post(url, {
            'appeal_id': str(self.appeal.id),
            'action': 'accept',
        })
        self.assertEqual(resp1.status_code, 302)
        self.appeal.refresh_from_db()
        self.violation.refresh_from_db()
        self.assertEqual(self.appeal.status, 'accepted')
        self.assertEqual(self.violation.is_appealed, 'accepted')
        self.assertEqual(self.appeal.officer, self.officer.officer_profile)
        self.assertFalse(Suspension.objects.exists())

        # Now reject a fresh appeal
        self.appeal = Appeal.objects.create(
            citizen=self.citizen.citizen_profile,
            violation=self.violation,
            reason='Still valid'
        )
        resp2 = self.client.post(url, {
            'appeal_id': str(self.appeal.id),
            'action': 'reject'
        })
        self.appeal.refresh_from_db()
        self.violation.refresh_from_db()
        self.assertEqual(self.appeal.status, 'rejected')
        self.assertEqual(self.violation.is_appealed, 'rejected')


class ReviewedAppealHistoryViewTests(TestCase):
    def setUp(self):
        # Officer + two handled appeals
        self.officer = CustomUser.objects.create_user(
            username='officer3', password='pass',
            user_type='officer', email='o3@example.com'
        )
        self.citizen = CustomUser.objects.create_user(
            username='citizen3', password='pass',
            user_type='citizen', email='c3@example.com'
        )
        mc = Motorcycle.objects.create(
            owner=self.citizen.citizen_profile,
            registration_number='LMN-0001', model_name='Bike3'
        )
        viol = Violation.objects.create(
            reporter=self.citizen.citizen_profile,
            motorcycle=mc,
            violation_type='racing',
            license_plate=mc.registration_number,
            cropped_image='dummy.jpg'
        )
        # Accepted & rejected
        self.a1 = Appeal.objects.create(
            citizen=self.citizen.citizen_profile,
            violation=viol,
            reason='Test',
            status='accepted',
            officer=self.officer.officer_profile
        )
        self.a2 = Appeal.objects.create(
            citizen=self.citizen.citizen_profile,
            violation=viol,
            reason='Test2',
            status='rejected',
            officer=self.officer.officer_profile
        )

        self.client = Client()
        self.client.login(username='officer3', password='pass')

    def test_reviewed_appeal_history(self):
        url = reverse('reviewed_appeal_history')
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        history = resp.context['reviewed_appeals']
        self.assertIn(self.a1, history)
        self.assertIn(self.a2, history)
