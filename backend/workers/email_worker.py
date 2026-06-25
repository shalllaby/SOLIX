import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve configuration from environment variables
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "solixagentic@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "qcamqwjxfggkqxss")
OTP_EXPIRE_MINUTES = 5

def get_html_content(username: str, otp: str) -> str:
    """Return a premium, responsive, dark-mode-themed HTML template for OTP verification."""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Verify Your Account</title>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background-color: #0b0f19;
                color: #e2e8f0;
            }}
            .container {{
                max-width: 600px;
                margin: 40px auto;
                background: linear-gradient(135deg, #111827 0%, #1e1b4b 100%);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
                text-align: center;
            }}
            .logo {{
                font-size: 28px;
                font-weight: 800;
                letter-spacing: 2px;
                background: linear-gradient(to right, #6366f1, #3b82f6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 30px;
            }}
            h1 {{
                font-size: 24px;
                color: #ffffff;
                margin-bottom: 10px;
                font-weight: 700;
            }}
            p {{
                font-size: 15px;
                line-height: 1.6;
                color: #94a3b8;
                margin-bottom: 25px;
            }}
            .otp-code {{
                display: inline-block;
                letter-spacing: 8px;
                font-size: 38px;
                font-weight: 800;
                color: #ffffff;
                background: rgba(99, 102, 241, 0.1);
                border: 1px solid rgba(99, 102, 241, 0.3);
                padding: 15px 35px;
                border-radius: 12px;
                margin: 20px 0;
                box-shadow: 0 0 20px rgba(99, 102, 241, 0.15);
            }}
            .warning {{
                font-size: 12px;
                color: #ef4444;
                margin-top: 25px;
                opacity: 0.8;
            }}
            .footer {{
                margin-top: 40px;
                font-size: 12px;
                color: #64748b;
                border-top: 1px solid rgba(255, 255, 255, 0.05);
                padding-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">SOLIX PLATFORM</div>
            <h1>Email Verification Needed</h1>
            <p>Hello <strong>{username}</strong>,</p>
            <p>Thank you for registering with SOLIX Platform. Please enter the secure 6-digit verification code below to activate your account. This code is valid for <strong>{OTP_EXPIRE_MINUTES} minutes</strong>.</p>
            
            <div class="otp-code">{otp}</div>
            
            <p class="warning">If you did not request this code, please ignore this email or contact support.</p>
            
            <div class="footer">
                &copy; 2026 SOLIX Platform. All rights reserved. Secure Standalone Authentication.
            </div>
        </div>
    </body>
    </html>
    """

def send_otp_email(recipient_email: str, username: str, otp: str) -> bool:
    """Send an HTML email containing the OTP via Gmail SMTP SSL connection (port 465)."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"SOLIX Platform Verification Code: {otp}"
        msg["From"] = SMTP_USER
        msg["To"] = recipient_email

        # Set Plain-text version and HTML version
        text = f"Hello {username},\n\nYour SOLIX Platform OTP verification code is: {otp}\nValid for {OTP_EXPIRE_MINUTES} minutes."
        html = get_html_content(username, otp)

        msg.attach(MIMEText(text, "plain"))
        msg.attach(MIMEText(html, "html"))

        logger.info(f"Connecting to SMTP server {SMTP_HOST}:{SMTP_PORT} via SSL...")
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            logger.info("Logging in to SMTP server...")
            server.login(SMTP_USER, SMTP_PASSWORD)
            logger.info(f"Sending email to {recipient_email}...")
            server.sendmail(SMTP_USER, recipient_email, msg.as_string())
            logger.info("Email sent successfully!")
            return True
            
    except Exception as e:
        logger.error(f"Failed to send email to {recipient_email}: {str(e)}", exc_info=True)
        return False
