# motorcycle_violation/management/commands/clear_users.py
from django.core.management.base import BaseCommand
from motorcycle_violation.models import CustomUser  # Adjust the import as necessary

class Command(BaseCommand):
    help = 'Clear all user data from the database.'

    def handle(self, *args, **kwargs):
        # Confirm deletion
        confirmation = input("Are you sure you want to delete all users? (yes/no): ")
        if confirmation.lower() != 'yes':
            self.stdout.write(self.style.WARNING('Operation cancelled. No users were deleted.'))
            return
        
        # Clear all users
        deleted_count, _ = CustomUser.objects.all().delete()
        self.stdout.write(self.style.SUCCESS(f'Successfully deleted {deleted_count} users.'))
