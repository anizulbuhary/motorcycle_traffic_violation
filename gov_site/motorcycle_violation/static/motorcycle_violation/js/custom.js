document.addEventListener('DOMContentLoaded', function () {
    const userTypeSelect = document.querySelector('#id_user_type');
    const cnicField = document.querySelector('#cnic-field');
    const employeeIdField = document.querySelector('#employee-id-field');

    function toggleFields() {
        if (userTypeSelect.value === 'citizen') {
            cnicField.style.display = 'block';  // Show CNIC field
            employeeIdField.style.display = 'none';  // Hide Employee ID field
        } else if (userTypeSelect.value === 'officer') {
            cnicField.style.display = 'none';  // Hide CNIC field
            employeeIdField.style.display = 'block';  // Show Employee ID field
        }
    }

    // Initialize fields on page load
    toggleFields();

    // Add event listener for user type change
    userTypeSelect.addEventListener('change', toggleFields);
});
