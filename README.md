
# Motorcycle Violation Government Web App

## Overview

The **Motorcycle Violation Government Web App** is a civic technology initiative aimed at enhancing road safety and compliance among motorcycle riders in Pakistan. This application enables citizens to report violations related to motorcycle usage effectively. Utilizing state-of-the-art computer vision technology, specifically YOLOv11, the app detects violations such as:

- **Riding without a helmet**
- **Carrying excessive passengers on a motorcycle**

Citizens who report violations will receive fines, and the app includes a functionality for appeals in cases of incorrect detections. Users can download payment receipts and make payments through designated banks. Furthermore, citizens who report violations will earn credits, which can be redeemed for various benefits.

### Features

- **Violation Detection**: Automatic detection of motorcycle violations using YOLOv8 technology.
- **Reporting Mechanism**: User-friendly interface for citizens to report violations effortlessly.
- **Fine System**: Citizens receive fines based on detected violations.
- **Appeals Process**: Users can appeal against wrongful detections, which are then reviewed by designated government officers.
- **Receipt Download**: Ability for users to download payment receipts for fines.
- **Payment Integration**: Secure payment processing through bank channels.
- **Credit System**: Citizens earn credits for reporting violations, which can be redeemed for rewards.
- **App Suspention System**: Officers can suspend citizens if they misuse the app..
- **Government Officer Role**: Separate user interface for government officers to manage and evaluate appeals.


## Installation

To set up the application on your local machine, follow the steps below:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/anizulbuhary/motorcycle_traffic_violation.git
   cd motorcycle_traffic_violation
   ```

2. **Install Requirements**:

   Ensure you have Python installed. Navigate to the project directory and install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install gobject**:

   The application requires the `gobject` library. To install it, you can use MSYS2 or your preferred package manager. Ensure the library is accessible in your environment.

Apologies for the misunderstanding! Here's the **4th step** rewritten exactly as you requested:

4. **Install Dependencies for Real-ESRGAN**:

   **Note**: **Real-ESRGAN** (Real Enhanced Super-Resolution Generative Adversarial Networks) is an advanced image upscaling tool designed to enhance image quality by improving resolution and restoring fine details in images. It uses a deep learning-based GAN model to upscale low-resolution images to higher resolutions while preserving their visual quality.

   - `py-real-esrgan 2.0.0`
   ```bash
   pip install py-real-esrgan
   ```

   - Installation
   ```bash
   pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
   ```

   - In case of an error related to `cached_download` cannot be imported, downgrade `huggingface_hub`:
   ```bash
   pip uninstall huggingface_hub
   ```

   - Then install `huggingface_hub` version `0.25.0`:
   ```bash
   pip install huggingface_hub==0.25.0
   ```


5. **Run the Application**:

   Start the development server with the following command:

   ```bash
   python manage.py runserver
   ```

   Access the application by navigating to `http://127.0.0.1:8000/` in your web browser.

You can now copy and paste this into your `README.md` file.

## Usage

- **For Citizens**:
  - **Register** for an account.
  - **Report Violations** using the reporting feature.
  - **Appeal Fines** if a violation was reported incorrectly.
  - **Manage Credits** earned from reporting violations and redeem them for benefits.
  
- **For Government Officers**:
  - **Review Appeals** submitted by citizens.
  - **Accept or Reject Appeals** based on manual checks and documentation.

## Requirements

- Python 3.x
- Django
- gobject library
- YOLOv11 for violation detection

## Troubleshooting

If you encounter issues while running the application, please check the following:

- Ensure all required libraries are installed as listed in `requirements.txt`.
- Verify that the `gobject` library is correctly installed and accessible in your environment.
- Check for any missing dependencies or updates required by the libraries in use.

For further assistance, refer to the [WeasyPrint Installation Documentation](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation) and [Troubleshooting Guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#troubleshooting).

   - In case of an error related to `cached_download` cannot be imported, downgrade `huggingface_hub`:
   ```bash
   pip uninstall huggingface_hub
   ```

   - Then install `huggingface_hub` version `0.25.0`:
   ```bash
   pip install huggingface_hub==0.25.0
   ```

## Contributing

Contributions are welcome! If you would like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch to your forked repository.
5. Submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For any inquiries, support, or feedback, please reach out to [anizulbuhary@gmail.com].

## Acknowledgments

- **YOLOv11** for its exceptional object detection capabilities.
- **Django** for its robust web application framework.
- All contributors and supporters of the project for their invaluable input.
- Special thanks to the open-source community for their resources and tools that made this project possible.
