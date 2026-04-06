from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.contrib.messages import get_messages

User = get_user_model()

class RegisterViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.register_url = reverse('register')
        self.login_url = reverse('login')

    def test_get_register_page(self):
        """
        GET to the register view should return status 200 and include the form in context.
        """
        response = self.client.get(self.register_url)
        self.assertEqual(response.status_code, 200)
        # The view sets form=None on GET
        self.assertIn('form', response.context)
        self.assertIsNone(response.context['form'])

    def test_post_valid_citizen_registration(self):
        """
        Posting valid citizen data should create a new user, profile,
        set a success message, and redirect to login.
        """
        data = {
            'user_type': 'citizen',
            'username': 'test_citizen',
            'password1': 'StrongPass123',
            'password2': 'StrongPass123',
            'password': 'StrongPass123',
            'email': 'citizen@example.com',
            'first_name': 'Test',
            'last_name': 'Citizen',
            'cnic': '1234567890123',  # 13-digit CNIC
        }
        response = self.client.post(self.register_url, data)
        # Should redirect to login page
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, self.login_url)

        # User created
        user = User.objects.get(username='test_citizen')
        self.assertEqual(user.email, 'citizen@example.com')
        self.assertEqual(user.user_type, 'citizen')
        # Profile created
        self.assertTrue(hasattr(user, 'citizen_profile'))

        # Success message
        storage = get_messages(response.wsgi_request)
        messages = [m.message for m in storage]
        self.assertTrue(any('Your account has been created successfully' in m for m in messages))

    def test_post_valid_officer_registration(self):
        """
        Posting valid officer data should create a new user, profile,
        set a success message, and redirect to login.
        """
        data = {
            'user_type': 'officer',
            'username': 'test_officer',
            'password1': 'OfficerPass123',
            'password2': 'OfficerPass123',
            'password': 'OfficerPass123',
            'email': 'officer@example.com',
            'first_name': 'Test',
            'last_name': 'Officer',
            'employee_id': '1234567890',  # 10-digit ID
        }
        response = self.client.post(self.register_url, data)
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, self.login_url)

        user = User.objects.get(username='test_officer')
        self.assertEqual(user.user_type, 'officer')
        self.assertTrue(hasattr(user, 'officer_profile'))

        storage = get_messages(response.wsgi_request)
        messages = [m.message for m in storage]
        self.assertTrue(any('Your account has been created successfully' in m for m in messages))

    def test_post_invalid_user_type(self):
        """
        Posting with an invalid user_type should redirect back to register and
        set an error message.
        """
        data = {
            'user_type': 'invalid_type',
            'username': 'should_fail',
            'password1': 'NoMatter123',
            'password2': 'NoMatter123',
            'password': 'NoMatter123',
            'email': 'fail@example.com',
            'first_name': 'Fail',
            'last_name': 'Case'
        }
        response = self.client.post(self.register_url, data)
        # Should redirect back to register
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, self.register_url)

        # No user created
        self.assertFalse(User.objects.filter(username='should_fail').exists())

        # Error message
        storage = get_messages(response.wsgi_request)
        messages = [m.message for m in storage]
        self.assertTrue(any('Invalid user type selected' in m for m in messages))

    def test_post_duplicate_email_registration(self):
        """
        Posting with an email that’s already registered should
        redirect back to register, not create a new user,
        and display an appropriate error message.
        """
        # Pre-create a user with the email we’ll reuse
        User.objects.create_user(
            username='existing_user',
            password='ExistingPass123',
            user_type='citizen',
            email='citizen@example.com',
            first_name='Existing',
            last_name='User'
        )

        data = {
            'user_type': 'citizen',
            'username': 'new_citizen',
            'password1': 'StrongPass123',
            'password2': 'StrongPass123',
            'password': 'StrongPass123',
            'email': 'citizen@example.com',  # duplicate
            'first_name': 'New',
            'last_name': 'Citizen',
            'cnic': '9876543210987',
        }
        response = self.client.post(self.register_url, data)

        # Ensure the registration form is re-rendered with errors
        self.assertEqual(response.status_code, 200)

        # Ensure no new user was created
        self.assertFalse(User.objects.filter(username='new_citizen').exists())

        form = response.context['form']  # get the form from the response context
        self.assertFormError(form, 'email', 'An account with this email already exists.')



class CitizenLoginTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.username = 'login_citizen'
        self.password = 'LoginPass123'
        self.user = User.objects.create_user(
            username=self.username,
            password=self.password,
            user_type='citizen',
            email='login@example.com',
            first_name='Login',
            last_name='Citizen'
        )
        self.login_url = reverse('login')

    def test_citizen_can_login(self):
        """
        A citizen with correct credentials should be able to log in.
        """
        logged_in = self.client.login(username=self.username, password=self.password)
        self.assertTrue(logged_in)

    def test_login_failure_wrong_password(self):
        """
        Login should fail with incorrect password.
        """
        result = self.client.login(username=self.username, password='WrongPassword')
        self.assertFalse(result)
